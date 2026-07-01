/**
* This file is part of xfeatSLAM.
*
* XFeat vocabulary support is backed by DBoW3 so float descriptors can be used
* without forcing ORB-style binary descriptors.
*/

#ifndef XFEATVOCABULARY_H
#define XFEATVOCABULARY_H

#include "thirdparty/DBoW3/src/DBoW3.h"

namespace ORB_SLAM3
{

typedef DBoW3::Vocabulary XFeatVocabulary;

} // namespace ORB_SLAM3

#endif // XFEATVOCABULARY_H
