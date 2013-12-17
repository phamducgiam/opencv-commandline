//
//  main.cpp
//  bow_generate
//
//  Created by Pham Duc Giam on 08/12/13.
//  Copyright (c) 2013 Pham Duc Giam. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

typedef enum {kLongOptionIndexNone, kLongOptionIndexDirectory, kLongOptionIndexDetector, kLongOptionIndexDetectorAdapter, kLongOptionIndexExtractor, kLongOptionIndexExtractorAdapter, kLongOptionIndexMatcher, kLongOptionIndexFeaturesOutput, kLongOptionIndexDescriptorsOutput} LongOptionIndex;

static const char* detectorAlgorithms[] = {"SURF","FAST","STAR","SIFT","ORB","BRISK","MSER","GFTT","HARRIS","Dense","SimpleBlob"};
static const char* extractorAlgorithms[] = {"SURF","SIFT","BRIEF","BRISK","ORB","FREAK"};
static const char* matchAlgorithms[] = {"FlannBased","BruteForce","BruteForce-L1","BruteForce-Hamming","BruteForce-Hamming(2)"};
static const char* featuresOutputDefault = "features.yml";
static const char* descriptorsOutputDefault = "descriptors.yml";

static const int BOW_TRAINER_RETRIES = 3;
static const int BOW_TRAINER_FLAGS = KMEANS_PP_CENTERS;

const Ptr<FeatureDetector> getDetector(const char *detectorAdapter, const char *detectorAlgorithm);
const Ptr<DescriptorExtractor> getExtractor(const char *extractorAdapter, const char *extractorAlgorithm);
const BOWKMeansTrainer getBOWTrainer(int vocabularySize);
const Ptr<DescriptorMatcher> getMatcher(const char *matchAlgorithm);

int main(int argc, char * const *argv)
{
    const char *directoryName, *detectorAlgorithm, *detectorAdapter, *extractorAlgorithm, *extractorAdapter, *matchAlgorithm, *featuresOutput, *descriptorsOutput;
    directoryName = detectorAlgorithm = detectorAdapter = extractorAlgorithm = extractorAdapter = matchAlgorithm = featuresOutput = descriptorsOutput = NULL;
    
    struct option longOptions[] = {
        {"directory", required_argument, 0, kLongOptionIndexDirectory},
        {"detector", required_argument, 0, kLongOptionIndexDetector},
        {"detector_adapter", required_argument, 0, kLongOptionIndexDetectorAdapter},
        {"extractor", required_argument, 0, kLongOptionIndexExtractor},
        {"extractor_adapter", required_argument, 0, kLongOptionIndexExtractorAdapter},
        {"matcher", required_argument, 0, kLongOptionIndexMatcher},
        {"features_output", required_argument, 0, kLongOptionIndexFeaturesOutput},
        {"descriptors_output", required_argument, 0, kLongOptionIndexDescriptorsOutput},
        {0, 0, 0, 0}
    };
    
    int c, optionIndex;
    while ((c=getopt_long(argc, argv, "", longOptions, &optionIndex))!=-1) {
        switch (c) {
            case kLongOptionIndexDirectory: {
                directoryName = optarg;
                break;
            }
            case kLongOptionIndexDetector: {
                detectorAlgorithm = optarg;
                break;
            }
            case kLongOptionIndexDetectorAdapter: {
                detectorAdapter = optarg;
                break;
            }
            case kLongOptionIndexExtractor: {
                extractorAlgorithm = optarg;
                break;
            }
            case kLongOptionIndexExtractorAdapter: {
                extractorAdapter = optarg;
                break;
            }
            case kLongOptionIndexMatcher: {
                matchAlgorithm = optarg;
                break;
            }
            case kLongOptionIndexFeaturesOutput: {
                featuresOutput = optarg;
                break;
            }
            case kLongOptionIndexDescriptorsOutput: {
                descriptorsOutput = optarg;
                break;
            }
            default:
                break;
        }
    }
    
    if (!directoryName) {
        cout << "need input directory" << endl;
        return -1;
    }
    
    if (!detectorAlgorithm) {
        cout << "use " << detectorAlgorithms[0] << " as detection algorithm" << endl;
        detectorAlgorithm = detectorAlgorithms[0];
    }
    
    if (!extractorAlgorithm) {
        cout << "use " << extractorAlgorithms[0] << " as extraction algorithm" << endl;
        extractorAlgorithm = extractorAlgorithms[0];
    }
    
    if (!matchAlgorithm) {
        cout << "use " << matchAlgorithms[0] << " as matcher algorithm" << endl;
        matchAlgorithm = matchAlgorithms[0];
    }
    
    if (!featuresOutput) {
        cout << "use " << featuresOutputDefault << " as output for features" << endl;
        featuresOutput = featuresOutputDefault;
    }
    if (!descriptorsOutput) {
        cout << "use " << descriptorsOutputDefault << " as output for descriptors" << endl;
        descriptorsOutput = descriptorsOutputDefault;
    }
    
    DIR *dir;
    dir = opendir(directoryName);
    if (!dir) {
        cout << "could not open directory " << directoryName << endl;
        return -1;
    }
    
    struct dirent *ep;
    char *filename;
    char path[1000];
    struct stat buf;
    
    int vocabularySize = 0;
    
    Ptr<FeatureDetector> detector = getDetector(detectorAdapter, detectorAlgorithm);
    Ptr<DescriptorExtractor> extractor = getExtractor(extractorAdapter, extractorAlgorithm);
    cv::vector<KeyPoint> keypoints;
    Mat image;
    Mat descriptors;
    Mat features;
    
    cout << "Building vocabulary..." << endl;
    while ((ep = readdir(dir))) {
    	filename = ep->d_name;
        if((strchr(filename, '.')-filename)==0)
        	continue;
        
        sprintf(path, "%s/%s", directoryName, filename);
        
        lstat(path, &buf);
        if (S_ISREG(buf.st_mode)) {
        	image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        	if (image.data) {
                cout << "File " << filename << "...";
        		detector->detect(image, keypoints);
        		extractor->compute(image, keypoints, descriptors);
             	features.push_back(descriptors);
                cout << " done" << endl;
             	vocabularySize++;
                //break;
            }
        }
    }
    closedir(dir);
    
    if (!vocabularySize) {
        cout << "There is no image in directory" << endl;
        return -2;
    }
    
    cout << "cluster features...";
    
    BOWKMeansTrainer bowTrainer = getBOWTrainer(vocabularySize);
    Mat vocabulary = bowTrainer.cluster(features);
    
    cout << "\tdone" << endl;
    
    cout << "write features to file " << featuresOutput << "...";
    FileStorage fsFeatures(featuresOutput, FileStorage::WRITE);
    fsFeatures << "vocabulary" << vocabulary;
    cout << "\tdone" << endl;
    fsFeatures.release();
    
    Ptr<DescriptorMatcher> matcher = getMatcher(matchAlgorithm);
    
    BOWImgDescriptorExtractor bowExtractor(extractor, matcher);
    bowExtractor.setVocabulary(vocabulary);
    
    cv::vector<Mat> bowDescriptors;
    Mat bowDescriptor;
    cv::vector<string> filenames;
    
    dir = opendir(directoryName);
    cout << "Generate bow descriptors..." << endl;
    while ((ep = readdir(dir))) {
    	filename = ep->d_name;
        if((strchr(filename, '.')-filename)==0)
        	continue;
        
        sprintf(path, "%s/%s", directoryName, filename);
        
        lstat(path, &buf);
        if (S_ISREG(buf.st_mode)) {
        	image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        	if (image.data) {
                cout << "File " << filename << "...";
        		detector->detect(image, keypoints);
        		bowExtractor.compute(image, keypoints, bowDescriptor);
                bowDescriptors.push_back(bowDescriptor);
                filenames.push_back(filename);
                cout << " done" << endl;
                //break;
            }
        }
    }
    closedir(dir);
    
    cout << "write descriptors to file " << descriptorsOutput << "...";
    FileStorage fsDescriptors(descriptorsOutput, FileStorage::WRITE);
    fsDescriptors << "descriptors" << bowDescriptors;
    fsDescriptors << "filenames" << filenames;
    cout << "\tdone" << endl;
    fsDescriptors.release();
    
    return 0;
}

const Ptr<FeatureDetector> getDetector(const char *detectorAdapter, const char *detectorAlgorithm)
{
    string detectorType = detectorAlgorithm;
    if (detectorAdapter) {
        detectorType = detectorAdapter + detectorType;
    }
    
    cout << "create feature detector with type: " << detectorType << endl;
    Ptr<FeatureDetector> result = FeatureDetector::create(detectorType);
    
    return result;
}

const Ptr<DescriptorExtractor> getExtractor(const char *extractorAdapter, const char *extractorAlgorithm)
{
    string extractorType = extractorAlgorithm;
    if (extractorAdapter) {
        extractorType = extractorAdapter + extractorType;
    }
    
    cout << "create descriptor extractor with type: " << extractorType << endl;
    Ptr<DescriptorExtractor> result = DescriptorExtractor::create(extractorType);
    
    return result;
}

const BOWKMeansTrainer getBOWTrainer(int vocabularySize)
{
    TermCriteria termCriteria(CV_TERMCRIT_ITER, 100, 0.001);
    BOWKMeansTrainer bowTrainer(vocabularySize, termCriteria, BOW_TRAINER_RETRIES, BOW_TRAINER_FLAGS);
    
    return bowTrainer;
    //Ptr<BOWKMeansTrainer> result(&bowTrainer);
    
    //return result;
}

const Ptr<DescriptorMatcher> getMatcher(const char *matchAlgorithm)
{
    return DescriptorMatcher::create(matchAlgorithm);
}

