#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/core/utility.hpp"
// #define __OPENCV_BUILD
// #include "opencv2/core/private.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/hal.hpp"

#include <algorithm>

#include <float.h>
#include <stdio.h>
#include "opencv2/core/hal/intrin.hpp"

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

#include "optflow_ws.h"

void calcScharrDeriv(const cv::Mat& src, cv::Mat& dst)
{
    using namespace cv;
    int rows = src.rows, cols = src.cols, cn = src.channels(), depth = src.depth();
    CV_Assert(depth == CV_8U);
    dst.create(rows, cols, CV_MAKETYPE(DataType<deriv_type>::depth, cn*2));
    parallel_for_(Range(0, rows), ScharrDerivInvoker(src, dst), cv::getNumThreads());
}

void ScharrDerivInvoker::operator()(const Range& range) const
{
    int rows = src.rows, cols = src.cols, cn = src.channels(), colsn = cols*cn;

    int x, y, delta = (int)alignSize((cols + 2)*cn, 16);
    AutoBuffer<deriv_type> _tempBuf(delta*2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf.data() + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);

#if CV_SIMD128
    v_int16x8 c3 = v_setall_s16(3), c10 = v_setall_s16(10);
#endif

    for( y = range.start; y < range.end; y++ )
    {
        const uchar* srow0 = src.ptr<uchar>(y > 0 ? y-1 : rows > 1 ? 1 : 0);
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow2 = src.ptr<uchar>(y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
        deriv_type* drow = (deriv_type *)dst.ptr<deriv_type>(y);

        // do vertical convolution
        x = 0;
#if CV_SIMD128
        {
            for( ; x <= colsn - 8; x += 8 )
            {
                v_int16x8 s0 = v_reinterpret_as_s16(v_load_expand(srow0 + x));
                v_int16x8 s1 = v_reinterpret_as_s16(v_load_expand(srow1 + x));
                v_int16x8 s2 = v_reinterpret_as_s16(v_load_expand(srow2 + x));

                v_int16x8 t1 = s2 - s0;
                v_int16x8 t0 = v_mul_wrap(s0 + s2, c3) + v_mul_wrap(s1, c10);

                v_store(trow0 + x, t0);
                v_store(trow1 + x, t1);
            }
        }
#endif

        for( ; x < colsn; x++ )
        {
            int t0 = (srow0[x] + srow2[x])*3 + srow1[x]*10;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        // make border
        int x0 = (cols > 1 ? 1 : 0)*cn, x1 = (cols > 1 ? cols-2 : 0)*cn;
        for( int k = 0; k < cn; k++ )
        {
            trow0[-cn + k] = trow0[x0 + k]; trow0[colsn + k] = trow0[x1 + k];
            trow1[-cn + k] = trow1[x0 + k]; trow1[colsn + k] = trow1[x1 + k];
        }

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;
#if CV_SIMD128
        {
            for( ; x <= colsn - 8; x += 8 )
            {
                v_int16x8 s0 = v_load(trow0 + x - cn);
                v_int16x8 s1 = v_load(trow0 + x + cn);
                v_int16x8 s2 = v_load(trow1 + x - cn);
                v_int16x8 s3 = v_load(trow1 + x);
                v_int16x8 s4 = v_load(trow1 + x + cn);

                v_int16x8 t0 = s1 - s0;
                v_int16x8 t1 = v_mul_wrap(s2 + s4, c3) + v_mul_wrap(s3, c10);

                v_store_interleave((drow + x*2), t0, t1);
            }
        }
#endif
        for( ; x < colsn; x++ )
        {
            deriv_type t0 = (deriv_type)(trow0[x+cn] - trow0[x-cn]);
            deriv_type t1 = (deriv_type)((trow1[x+cn] + trow1[x-cn])*3 + trow1[x]*10);
            drow[x*2] = t0; drow[x*2+1] = t1;
        }
    }
}

LKTrackerInvoker::LKTrackerInvoker(
                      const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                      const Point2f* _prevPts, Point2f* _nextPts,
                      uchar* _status, float* _err,
                      Size _winSize, TermCriteria _criteria,
                      int _level, int _maxLevel, int _flags, float _minEigThreshold )
{
    prevImg = &_prevImg;
    prevDeriv = &_prevDeriv;
    nextImg = &_nextImg;
    prevPts = _prevPts;
    nextPts = _nextPts;
    status = _status;
    err = _err;
    winSize = _winSize;
    criteria = _criteria;
    level = _level;
    maxLevel = _maxLevel;
    flags = _flags;
    minEigThreshold = _minEigThreshold;
}


void OptflowWS::calc(InputArray _prevImg, InputArray _nextImg,
                        InputArray _prevPts, InputOutputArray _nextPts,
                        OutputArray _status, OutputArray _err)
{
    Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<deriv_type>::depth;

    CV_Assert( maxLevel >= 0 && winSize.width > 2 && winSize.height > 2 );

    int level=0, i, npoints;
    CV_Assert( (npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0 );

    if( npoints == 0 )
    {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }

    if( !(flags & OPTFLOW_USE_INITIAL_FLOW) )
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    Mat nextPtsMat = _nextPts.getMat();
    CV_Assert( nextPtsMat.checkVector(2, CV_32F, true) == npoints );

    const Point2f* prevPts = prevPtsMat.ptr<Point2f>();
    Point2f* nextPts = nextPtsMat.ptr<Point2f>();

    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert( statusMat.isContinuous() );
    uchar* status = statusMat.ptr();
    float* err = 0;

    for( i = 0; i < npoints; i++ )
        status[i] = true;

    if( _err.needed() )
    {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert( errMat.isContinuous() );
        err = errMat.ptr<float>();
    }

    std::vector<Mat> prevPyr, nextPyr;
    int levels1 = -1;
    int lvlStep1 = 1;
    int levels2 = -1;
    int lvlStep2 = 1;

    if(_prevImg.kind() == _InputArray::STD_VECTOR_MAT)
    {
        _prevImg.getMatVector(prevPyr);

        levels1 = int(prevPyr.size()) - 1;
        CV_Assert(levels1 >= 0);

        if (levels1 % 2 == 1 && prevPyr[0].channels() * 2 == prevPyr[1].channels() && prevPyr[1].depth() == derivDepth)
        {
            lvlStep1 = 2;
            levels1 /= 2;
        }

        // ensure that pyramid has required padding
        if(levels1 > 0)
        {
            Size fullSize;
            Point ofs;
            prevPyr[lvlStep1].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize.width && ofs.y >= winSize.height
                && ofs.x + prevPyr[lvlStep1].cols + winSize.width <= fullSize.width
                && ofs.y + prevPyr[lvlStep1].rows + winSize.height <= fullSize.height);
        }

        if(levels1 < maxLevel)
            maxLevel = levels1;
    }

    if(_nextImg.kind() == _InputArray::STD_VECTOR_MAT)
    {
        _nextImg.getMatVector(nextPyr);

        levels2 = int(nextPyr.size()) - 1;
        CV_Assert(levels2 >= 0);

        if (levels2 % 2 == 1 && nextPyr[0].channels() * 2 == nextPyr[1].channels() && nextPyr[1].depth() == derivDepth)
        {
            lvlStep2 = 2;
            levels2 /= 2;
        }

        // ensure that pyramid has required padding
        if(levels2 > 0)
        {
            Size fullSize;
            Point ofs;
            nextPyr[lvlStep2].locateROI(fullSize, ofs);
            CV_Assert(ofs.x >= winSize.width && ofs.y >= winSize.height
                && ofs.x + nextPyr[lvlStep2].cols + winSize.width <= fullSize.width
                && ofs.y + nextPyr[lvlStep2].rows + winSize.height <= fullSize.height);
        }

        if(levels2 < maxLevel)
            maxLevel = levels2;
    }

    if (levels1 < 0)
        maxLevel = buildOpticalFlowPyramid(_prevImg, prevPyr, winSize, maxLevel, false);

    if (levels2 < 0)
        maxLevel = buildOpticalFlowPyramid(_nextImg, nextPyr, winSize, maxLevel, false);

    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;

    // dI/dx ~ Ix, dI/dy ~ Iy
    Mat derivIBuf;
    if(lvlStep1 == 1)
        derivIBuf.create(prevPyr[0].rows + winSize.height*2, prevPyr[0].cols + winSize.width*2, CV_MAKETYPE(derivDepth, prevPyr[0].channels() * 2));

    for( level = maxLevel; level >= 0; level-- )
    {
        Mat derivI;
        if(lvlStep1 == 1)
        {
            Size imgSize = prevPyr[level * lvlStep1].size();
            Mat _derivI( imgSize.height + winSize.height*2,
                imgSize.width + winSize.width*2, derivIBuf.type(), derivIBuf.ptr() );
            derivI = _derivI(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
            calcScharrDeriv(prevPyr[level * lvlStep1], derivI);
            copyMakeBorder(derivI, _derivI, winSize.height, winSize.height, winSize.width, winSize.width, BORDER_CONSTANT|BORDER_ISOLATED);
        }
        else
            derivI = prevPyr[level * lvlStep1 + 1];

        CV_Assert(prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size());
        CV_Assert(prevPyr[level * lvlStep1].type() == nextPyr[level * lvlStep2].type());

        parallel_for_(Range(0, npoints), LKTrackerInvoker(prevPyr[level * lvlStep1], derivI,
                                                          nextPyr[level * lvlStep2], prevPts, nextPts,
                                                          status, err,
                                                          winSize, criteria, level, maxLevel,
                                                          flags, (float)minEigThreshold));
    }
}

void LKTrackerInvoker::operator()(const Range& range) const
{

    Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
    const Mat& I = *prevImg;
    const Mat& J = *nextImg;
    const Mat& derivI = *prevDeriv;

    int j, cn = I.channels(), cn2 = cn*2;
    cv::AutoBuffer<deriv_type> _buf(winSize.area()*(cn + cn2));
    int derivDepth = DataType<deriv_type>::depth;

    Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), _buf.data());
    Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2), _buf.data() + winSize.area()*cn);

    for( int ptidx = range.start; ptidx < range.end; ptidx++ )
    {
        Point2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
        Point2f nextPt;
        if( level == maxLevel )
        {
            if( flags & OPTFLOW_USE_INITIAL_FLOW )
                nextPt = nextPts[ptidx]*(float)(1./(1 << level));
            else
                nextPt = prevPt;
        }
        else
            nextPt = nextPts[ptidx]*2.f;
        nextPts[ptidx] = nextPt;

        Point2i iprevPt, inextPt;
        prevPt -= halfWin;
        iprevPt.x = cvFloor(prevPt.x);
        iprevPt.y = cvFloor(prevPt.y);

        if( iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
            iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows )
        {
            if( level == 0 )
            {
                if( status )
                    status[ptidx] = false;
                if( err )
                    err[ptidx] = 0;
            }
            continue;
        }

        float a = prevPt.x - iprevPt.x;
        float b = prevPt.y - iprevPt.y;
        const int W_BITS = 14, W_BITS1 = 14;
        const float FLT_SCALE = 1.f/(1 << 20);
        int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
        int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
        int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
        int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        int dstep = (int)(derivI.step/derivI.elemSize1());
        int stepI = (int)(I.step/I.elemSize1());
        int stepJ = (int)(J.step/J.elemSize1());
        float iA11 = 0, iA12 = 0, iA22 = 0;
        float A11, A12, A22;

#if CV_SIMD128 && !CV_NEON
        v_int16x8 qw0((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
        v_int16x8 qw1((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
        v_int32x4 qdelta_d = v_setall_s32(1 << (W_BITS1-1));
        v_int32x4 qdelta = v_setall_s32(1 << (W_BITS1-5-1));
        v_float32x4 qA11 = v_setzero_f32(), qA12 = v_setzero_f32(), qA22 = v_setzero_f32();
#endif

#if CV_NEON

        float CV_DECL_ALIGNED(16) nA11[] = { 0, 0, 0, 0 }, nA12[] = { 0, 0, 0, 0 }, nA22[] = { 0, 0, 0, 0 };
        const int shifter1 = -(W_BITS - 5); //negative so it shifts right
        const int shifter2 = -(W_BITS);

        const int16x4_t d26 = vdup_n_s16((int16_t)iw00);
        const int16x4_t d27 = vdup_n_s16((int16_t)iw01);
        const int16x4_t d28 = vdup_n_s16((int16_t)iw10);
        const int16x4_t d29 = vdup_n_s16((int16_t)iw11);
        const int32x4_t q11 = vdupq_n_s32((int32_t)shifter1);
        const int32x4_t q12 = vdupq_n_s32((int32_t)shifter2);

#endif

        // extract the patch from the first image, compute covariation matrix of derivatives
        int x, y;
        for( y = 0; y < winSize.height; y++ )
        {
            const uchar* src = I.ptr() + (y + iprevPt.y)*stepI + iprevPt.x*cn;
            const deriv_type* dsrc = derivI.ptr<deriv_type>() + (y + iprevPt.y)*dstep + iprevPt.x*cn2;

            deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
            deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

            x = 0;

#if CV_SIMD128 && !CV_NEON
            for( ; x <= winSize.width*cn - 8; x += 8, dsrc += 8*2, dIptr += 8*2 )
            {
                v_int32x4 t0, t1;
                v_int16x8 v00, v01, v10, v11, t00, t01, t10, t11;

                v00 = v_reinterpret_as_s16(v_load_expand(src + x));
                v01 = v_reinterpret_as_s16(v_load_expand(src + x + cn));
                v10 = v_reinterpret_as_s16(v_load_expand(src + x + stepI));
                v11 = v_reinterpret_as_s16(v_load_expand(src + x + stepI + cn));

                v_zip(v00, v01, t00, t01);
                v_zip(v10, v11, t10, t11);

                t0 = v_dotprod(t00, qw0, qdelta) + v_dotprod(t10, qw1);
                t1 = v_dotprod(t01, qw0, qdelta) + v_dotprod(t11, qw1);
                t0 = t0 >> (W_BITS1-5);
                t1 = t1 >> (W_BITS1-5);
                v_store(Iptr + x, v_pack(t0, t1));

                v00 = v_reinterpret_as_s16(v_load(dsrc));
                v01 = v_reinterpret_as_s16(v_load(dsrc + cn2));
                v10 = v_reinterpret_as_s16(v_load(dsrc + dstep));
                v11 = v_reinterpret_as_s16(v_load(dsrc + dstep + cn2));

                v_zip(v00, v01, t00, t01);
                v_zip(v10, v11, t10, t11);

                t0 = v_dotprod(t00, qw0, qdelta_d) + v_dotprod(t10, qw1);
                t1 = v_dotprod(t01, qw0, qdelta_d) + v_dotprod(t11, qw1);
                t0 = t0 >> W_BITS1;
                t1 = t1 >> W_BITS1;
                v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
                v_store(dIptr, v00);

                v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(v00))));
                v_expand(v00, t1, t0);

                v_float32x4 fy = v_cvt_f32(t0);
                v_float32x4 fx = v_cvt_f32(t1);

                qA22 = v_muladd(fy, fy, qA22);
                qA12 = v_muladd(fx, fy, qA12);
                qA11 = v_muladd(fx, fx, qA11);

                v00 = v_reinterpret_as_s16(v_load(dsrc + 4*2));
                v01 = v_reinterpret_as_s16(v_load(dsrc + 4*2 + cn2));
                v10 = v_reinterpret_as_s16(v_load(dsrc + 4*2 + dstep));
                v11 = v_reinterpret_as_s16(v_load(dsrc + 4*2 + dstep + cn2));

                v_zip(v00, v01, t00, t01);
                v_zip(v10, v11, t10, t11);

                t0 = v_dotprod(t00, qw0, qdelta_d) + v_dotprod(t10, qw1);
                t1 = v_dotprod(t01, qw0, qdelta_d) + v_dotprod(t11, qw1);
                t0 = t0 >> W_BITS1;
                t1 = t1 >> W_BITS1;
                v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
                v_store(dIptr + 4*2, v00);

                v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(v00))));
                v_expand(v00, t1, t0);

                fy = v_cvt_f32(t0);
                fx = v_cvt_f32(t1);

                qA22 = v_muladd(fy, fy, qA22);
                qA12 = v_muladd(fx, fy, qA12);
                qA11 = v_muladd(fx, fx, qA11);
            }
#endif

#if CV_NEON
            for( ; x <= winSize.width*cn - 4; x += 4, dsrc += 4*2, dIptr += 4*2 )
            {

                uint8x8_t d0 = vld1_u8(&src[x]);
                uint8x8_t d2 = vld1_u8(&src[x+cn]);
                uint16x8_t q0 = vmovl_u8(d0);
                uint16x8_t q1 = vmovl_u8(d2);

                int32x4_t q5 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26);
                int32x4_t q6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27);

                uint8x8_t d4 = vld1_u8(&src[x + stepI]);
                uint8x8_t d6 = vld1_u8(&src[x + stepI + cn]);
                uint16x8_t q2 = vmovl_u8(d4);
                uint16x8_t q3 = vmovl_u8(d6);

                int32x4_t q7 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28);
                int32x4_t q8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29);

                q5 = vaddq_s32(q5, q6);
                q7 = vaddq_s32(q7, q8);
                q5 = vaddq_s32(q5, q7);

                int16x4x2_t d0d1 = vld2_s16(dsrc);
                int16x4x2_t d2d3 = vld2_s16(&dsrc[cn2]);

                q5 = vqrshlq_s32(q5, q11);

                int32x4_t q4 = vmull_s16(d0d1.val[0], d26);
                q6 = vmull_s16(d0d1.val[1], d26);

                int16x4_t nd0 = vmovn_s32(q5);

                q7 = vmull_s16(d2d3.val[0], d27);
                q8 = vmull_s16(d2d3.val[1], d27);

                vst1_s16(&Iptr[x], nd0);

                int16x4x2_t d4d5 = vld2_s16(&dsrc[dstep]);
                int16x4x2_t d6d7 = vld2_s16(&dsrc[dstep+cn2]);

                q4 = vaddq_s32(q4, q7);
                q6 = vaddq_s32(q6, q8);

                q7 = vmull_s16(d4d5.val[0], d28);
                int32x4_t q14 = vmull_s16(d4d5.val[1], d28);
                q8 = vmull_s16(d6d7.val[0], d29);
                int32x4_t q15 = vmull_s16(d6d7.val[1], d29);

                q7 = vaddq_s32(q7, q8);
                q14 = vaddq_s32(q14, q15);

                q4 = vaddq_s32(q4, q7);
                q6 = vaddq_s32(q6, q14);

                float32x4_t nq0 = vld1q_f32(nA11);
                float32x4_t nq1 = vld1q_f32(nA12);
                float32x4_t nq2 = vld1q_f32(nA22);

                q4 = vqrshlq_s32(q4, q12);
                q6 = vqrshlq_s32(q6, q12);

                q7 = vmulq_s32(q4, q4);
                q8 = vmulq_s32(q4, q6);
                q15 = vmulq_s32(q6, q6);

                nq0 = vaddq_f32(nq0, vcvtq_f32_s32(q7));
                nq1 = vaddq_f32(nq1, vcvtq_f32_s32(q8));
                nq2 = vaddq_f32(nq2, vcvtq_f32_s32(q15));

                vst1q_f32(nA11, nq0);
                vst1q_f32(nA12, nq1);
                vst1q_f32(nA22, nq2);

                int16x4_t d8 = vmovn_s32(q4);
                int16x4_t d12 = vmovn_s32(q6);

                int16x4x2_t d8d12;
                d8d12.val[0] = d8; d8d12.val[1] = d12;
                vst2_s16(dIptr, d8d12);
            }
#endif

            for( ; x < winSize.width*cn; x++, dsrc += 2, dIptr += 2 )
            {
                int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                      src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1-5);
                int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                       dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
                int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                                       dsrc[dstep+cn2+1]*iw11, W_BITS1);

                Iptr[x] = (short)ival;
                dIptr[0] = (short)ixval;
                dIptr[1] = (short)iyval;

                iA11 += (float)(ixval*ixval);
                iA12 += (float)(ixval*iyval);
                iA22 += (float)(iyval*iyval);
            }
        }

#if CV_SIMD128 && !CV_NEON
        iA11 += v_reduce_sum(qA11);
        iA12 += v_reduce_sum(qA12);
        iA22 += v_reduce_sum(qA22);
#endif

#if CV_NEON
        iA11 += nA11[0] + nA11[1] + nA11[2] + nA11[3];
        iA12 += nA12[0] + nA12[1] + nA12[2] + nA12[3];
        iA22 += nA22[0] + nA22[1] + nA22[2] + nA22[3];
#endif

        A11 = iA11*FLT_SCALE;
        A12 = iA12*FLT_SCALE;
        A22 = iA22*FLT_SCALE;

        float D = A11*A22 - A12*A12;
        float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                        4.f*A12*A12))/(2*winSize.width*winSize.height);

        if( err && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0 )
            err[ptidx] = (float)minEig;

        if( minEig < minEigThreshold || D < FLT_EPSILON )
        {
            if( level == 0 && status )
                status[ptidx] = false;
            continue;
        }

        D = 1.f/D;

        nextPt -= halfWin;
        Point2f prevDelta;

        for( j = 0; j < criteria.maxCount; j++ )
        {
            inextPt.x = cvFloor(nextPt.x);
            inextPt.y = cvFloor(nextPt.y);

            if( inextPt.x < -winSize.width || inextPt.x >= J.cols ||
               inextPt.y < -winSize.height || inextPt.y >= J.rows )
            {
                if( level == 0 && status )
                    status[ptidx] = false;
                break;
            }

            a = nextPt.x - inextPt.x;
            b = nextPt.y - inextPt.y;
            iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            float ib1 = 0, ib2 = 0;
            float b1, b2;
#if CV_SIMD128 && !CV_NEON
            qw0 = v_int16x8((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
            qw1 = v_int16x8((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
            v_float32x4 qb0 = v_setzero_f32(), qb1 = v_setzero_f32();
#endif

#if CV_NEON
            float CV_DECL_ALIGNED(16) nB1[] = { 0,0,0,0 }, nB2[] = { 0,0,0,0 };

            const int16x4_t d26_2 = vdup_n_s16((int16_t)iw00);
            const int16x4_t d27_2 = vdup_n_s16((int16_t)iw01);
            const int16x4_t d28_2 = vdup_n_s16((int16_t)iw10);
            const int16x4_t d29_2 = vdup_n_s16((int16_t)iw11);

#endif

            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* Jptr = J.ptr() + (y + inextPt.y)*stepJ + inextPt.x*cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);
                const deriv_type* dIptr = derivIWinBuf.ptr<deriv_type>(y);

                x = 0;

#if CV_SIMD128 && !CV_NEON
                for( ; x <= winSize.width*cn - 8; x += 8, dIptr += 8*2 )
                {
                    v_int16x8 diff0 = v_reinterpret_as_s16(v_load(Iptr + x)), diff1, diff2;
                    v_int16x8 v00 = v_reinterpret_as_s16(v_load_expand(Jptr + x));
                    v_int16x8 v01 = v_reinterpret_as_s16(v_load_expand(Jptr + x + cn));
                    v_int16x8 v10 = v_reinterpret_as_s16(v_load_expand(Jptr + x + stepJ));
                    v_int16x8 v11 = v_reinterpret_as_s16(v_load_expand(Jptr + x + stepJ + cn));

                    v_int32x4 t0, t1;
                    v_int16x8 t00, t01, t10, t11;
                    v_zip(v00, v01, t00, t01);
                    v_zip(v10, v11, t10, t11);

                    t0 = v_dotprod(t00, qw0, qdelta) + v_dotprod(t10, qw1);
                    t1 = v_dotprod(t01, qw0, qdelta) + v_dotprod(t11, qw1);
                    t0 = t0 >> (W_BITS1-5);
                    t1 = t1 >> (W_BITS1-5);
                    diff0 = v_pack(t0, t1) - diff0;
                    v_zip(diff0, diff0, diff2, diff1); // It0 It0 It1 It1 ...
                    v00 = v_reinterpret_as_s16(v_load(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                    v01 = v_reinterpret_as_s16(v_load(dIptr + 8));
                    v_zip(v00, v01, v10, v11);
                    v_zip(diff2, diff1, v00, v01);
                    qb0 += v_cvt_f32(v_dotprod(v00, v10));
                    qb1 += v_cvt_f32(v_dotprod(v01, v11));
                }
#endif

#if CV_NEON
                for( ; x <= winSize.width*cn - 8; x += 8, dIptr += 8*2 )
                {

                    uint8x8_t d0 = vld1_u8(&Jptr[x]);
                    uint8x8_t d2 = vld1_u8(&Jptr[x+cn]);
                    uint8x8_t d4 = vld1_u8(&Jptr[x+stepJ]);
                    uint8x8_t d6 = vld1_u8(&Jptr[x+stepJ+cn]);

                    uint16x8_t q0 = vmovl_u8(d0);
                    uint16x8_t q1 = vmovl_u8(d2);
                    uint16x8_t q2 = vmovl_u8(d4);
                    uint16x8_t q3 = vmovl_u8(d6);

                    int32x4_t nq4 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q0)), d26_2);
                    int32x4_t nq5 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q0)), d26_2);

                    int32x4_t nq6 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q1)), d27_2);
                    int32x4_t nq7 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q1)), d27_2);

                    int32x4_t nq8 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q2)), d28_2);
                    int32x4_t nq9 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q2)), d28_2);

                    int32x4_t nq10 = vmull_s16(vget_low_s16(vreinterpretq_s16_u16(q3)), d29_2);
                    int32x4_t nq11 = vmull_s16(vget_high_s16(vreinterpretq_s16_u16(q3)), d29_2);

                    nq4 = vaddq_s32(nq4, nq6);
                    nq5 = vaddq_s32(nq5, nq7);
                    nq8 = vaddq_s32(nq8, nq10);
                    nq9 = vaddq_s32(nq9, nq11);

                    int16x8_t q6 = vld1q_s16(&Iptr[x]);

                    nq4 = vaddq_s32(nq4, nq8);
                    nq5 = vaddq_s32(nq5, nq9);

                    nq8 = vmovl_s16(vget_high_s16(q6));
                    nq6 = vmovl_s16(vget_low_s16(q6));

                    nq4 = vqrshlq_s32(nq4, q11);
                    nq5 = vqrshlq_s32(nq5, q11);

                    int16x8x2_t q0q1 = vld2q_s16(dIptr);
                    float32x4_t nB1v = vld1q_f32(nB1);
                    float32x4_t nB2v = vld1q_f32(nB2);

                    nq4 = vsubq_s32(nq4, nq6);
                    nq5 = vsubq_s32(nq5, nq8);

                    int32x4_t nq2 = vmovl_s16(vget_low_s16(q0q1.val[0]));
                    int32x4_t nq3 = vmovl_s16(vget_high_s16(q0q1.val[0]));

                    nq7 = vmovl_s16(vget_low_s16(q0q1.val[1]));
                    nq8 = vmovl_s16(vget_high_s16(q0q1.val[1]));

                    nq9 = vmulq_s32(nq4, nq2);
                    nq10 = vmulq_s32(nq5, nq3);

                    nq4 = vmulq_s32(nq4, nq7);
                    nq5 = vmulq_s32(nq5, nq8);

                    nq9 = vaddq_s32(nq9, nq10);
                    nq4 = vaddq_s32(nq4, nq5);

                    nB1v = vaddq_f32(nB1v, vcvtq_f32_s32(nq9));
                    nB2v = vaddq_f32(nB2v, vcvtq_f32_s32(nq4));

                    vst1q_f32(nB1, nB1v);
                    vst1q_f32(nB2, nB2v);
                }
#endif

                for( ; x < winSize.width*cn; x++, dIptr += 2 )
                {
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    ib1 += (float)(diff*dIptr[0]);
                    ib2 += (float)(diff*dIptr[1]);
                }
            }

#if CV_SIMD128 && !CV_NEON
            v_float32x4 qf0, qf1;
            v_recombine(v_interleave_pairs(qb0 + qb1), v_setzero_f32(), qf0, qf1);
            ib1 += v_reduce_sum(qf0);
            ib2 += v_reduce_sum(qf1);
#endif

#if CV_NEON

            ib1 += (float)(nB1[0] + nB1[1] + nB1[2] + nB1[3]);
            ib2 += (float)(nB2[0] + nB2[1] + nB2[2] + nB2[3]);
#endif

            b1 = ib1*FLT_SCALE;
            b2 = ib2*FLT_SCALE;

            Point2f delta( (float)((A12*b2 - A22*b1) * D),
                          (float)((A12*b1 - A11*b2) * D));
            //delta = -delta;

            nextPt += delta;
            nextPts[ptidx] = nextPt + halfWin;

            if( delta.ddot(delta) <= criteria.epsilon )
                break;

            if( j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
               std::abs(delta.y + prevDelta.y) < 0.01 )
            {
                nextPts[ptidx] -= delta*0.5f;
                break;
            }
            prevDelta = delta;
        }

        CV_Assert(status != NULL);
        if( status[ptidx] && err && level == 0 && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) == 0 )
        {
            Point2f nextPoint = nextPts[ptidx] - halfWin;
            Point inextPoint;

            inextPoint.x = cvFloor(nextPoint.x);
            inextPoint.y = cvFloor(nextPoint.y);

            if( inextPoint.x < -winSize.width || inextPoint.x >= J.cols ||
                inextPoint.y < -winSize.height || inextPoint.y >= J.rows )
            {
                if( status )
                    status[ptidx] = false;
                continue;
            }

            float aa = nextPoint.x - inextPoint.x;
            float bb = nextPoint.y - inextPoint.y;
            iw00 = cvRound((1.f - aa)*(1.f - bb)*(1 << W_BITS));
            iw01 = cvRound(aa*(1.f - bb)*(1 << W_BITS));
            iw10 = cvRound((1.f - aa)*bb*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            float errval = 0.f;

            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* Jptr = J.ptr() + (y + inextPoint.y)*stepJ + inextPoint.x*cn;
                const deriv_type* Iptr = IWinBuf.ptr<deriv_type>(y);

                for( x = 0; x < winSize.width*cn; x++ )
                {
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    errval += std::abs((float)diff);
                }
            }
            err[ptidx] = errval * 1.f/(32*winSize.width*cn*winSize.height);
        }
    }
}