#ifndef OPTFLOW_WS_H
#define OPTFLOW_WS_H

#include <iostream>
#include <queue>
#include <opencv2/opencv.hpp>

using namespace cv;

class OptflowWS
{
    struct dim3
    {
        unsigned int x, y, z;
        dim3() : x(0), y(0), z(0) { }
    };
    public:
        Size winSize;
        OptflowWS(Size winSize_ = Size(21,21),
                         int maxLevel_ = 3,
                         TermCriteria criteria_ = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                         int flags_ = 0,
                         double minEigThreshold_ = 1e-4) :
            winSize(winSize_), maxLevel(maxLevel_), criteria(criteria_), flags(flags_), minEigThreshold(minEigThreshold_)
        {
        }

        Size getWinSize() { return winSize;}
        void setWinSize(Size winSize_) { winSize = winSize_;}

        int getMaxLevel() { return maxLevel;}
        void setMaxLevel(int maxLevel_) { maxLevel = maxLevel_;}

        TermCriteria getTermCriteria() { return criteria;}
        void setTermCriteria(cv::TermCriteria& crit_) { criteria=crit_;}

        int getFlags() { return flags; }
        void setFlags(int flags_) { flags=flags_;}

        double getMinEigThreshold() { return minEigThreshold;}
        void setMinEigThreshold(double minEigThreshold_) { minEigThreshold=minEigThreshold_;}

        void calc(InputArray prevImg, InputArray nextImg,
                          InputArray prevPts, InputOutputArray nextPts,
                          OutputArray status,
                          OutputArray err = cv::noArray());

        String getDefaultName() { return "SparseOpticalFlow.SparsePyrLKOpticalFlow"; }

    private:
        int maxLevel;
        cv::TermCriteria criteria;
        int flags;
        double minEigThreshold;
};


typedef short deriv_type;

struct ScharrDerivInvoker : ParallelLoopBody
{
    ScharrDerivInvoker(const Mat& _src, const Mat& _dst)
        : src(_src), dst(_dst)
    { }

    void operator()(const Range& range) const CV_OVERRIDE;

    const Mat& src;
    const Mat& dst;
};

struct LKTrackerInvoker : ParallelLoopBody
{
    LKTrackerInvoker( const Mat& _prevImg, const Mat& _prevDeriv, const Mat& _nextImg,
                        const Point2f* _prevPts, Point2f* _nextPts,
                        uchar* _status, float* _err,
                        Size _winSize, TermCriteria _criteria,
                        int _level, int _maxLevel, int _flags, float _minEigThreshold );

    void operator()(const Range& range) const CV_OVERRIDE;

    const Mat* prevImg;
    const Mat* nextImg;
    const Mat* prevDeriv;
    const Point2f* prevPts;
    Point2f* nextPts;
    uchar* status;
    float* err;
    Size winSize;
    TermCriteria criteria;
    int level;
    int maxLevel;
    int flags;
    float minEigThreshold;
};

#endif