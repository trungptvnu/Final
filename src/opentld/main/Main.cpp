/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/
/*
 * MainX.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#include <iostream>
#include <cstdlib>
#include <pthread.h>


#include "Main.h"

#include "Config.h"
#include "ImAcq.h"
#include "Gui.h"
#include "TLDUtil.h"
#include "Trajectory.h"

using namespace std;
using namespace tld;
using namespace cv;

void Main::doWork()
{
	Trajectory trajectory;

    IplImage *img = imAcqGetImg(imAcq);

    Mat grey(img->height, img->width, CV_8UC1);

    cvtColor(cvarrToMat(img), grey, CV_BGR2GRAY);

    tld->detectorCascade->imgWidth = grey.cols;
    tld->detectorCascade->imgHeight = grey.rows;
    tld->detectorCascade->imgWidthStep = grey.step;

    _tld->detectorCascade->imgWidth = grey.cols;
    _tld->detectorCascade->imgHeight = grey.rows;
    _tld->detectorCascade->imgWidthStep = grey.step;

    _tld1->detectorCascade->imgWidth = grey.cols;
    _tld1->detectorCascade->imgHeight = grey.rows;
    _tld1->detectorCascade->imgWidthStep = grey.step;




	if(showTrajectory)
	{
		trajectory.init(trajectoryLength);
	}

    if(selectManually)
    {

        CvRect box;

        if(getBBFromUser(img, box, gui) == PROGRAM_EXIT)
        {
            return;
        }

        if(initialBB == NULL)
        {
            initialBB = new int[4];
        }

        initialBB[0] = box.x;
        initialBB[1] = box.y;
        initialBB[2] = box.width;
        initialBB[3] = box.height;
        
        // Lan 2

        CvRect box1;

        if(getBBFromUser(img, box1, gui) == PROGRAM_EXIT)
        {
            return;
        }

        if(initialBB1 == NULL)
        {
            initialBB1 = new int[4];
        }

        initialBB1[0] = box1.x;
        initialBB1[1] = box1.y;
        initialBB1[2] = box1.width;
        initialBB1[3] = box1.height;

        // Lan 3

        CvRect box2;

        if(getBBFromUser(img, box2, gui) == PROGRAM_EXIT)
        {
            return;
        }

        if(initialBB2 == NULL)
        {
            initialBB2 = new int[4];
        }

        initialBB2[0] = box2.x;
        initialBB2[1] = box2.y;
        initialBB2[2] = box2.width;
        initialBB2[3] = box2.height;


    }

    FILE *resultsFile = NULL;

    if(printResults != NULL)
    {
        resultsFile = fopen(printResults, "w");
        if(!resultsFile)
        {
            fprintf(stderr, "Error: Unable to create results-file \"%s\"\n", printResults);
            exit(-1);
        }
    }

    bool reuseFrameOnce = false;
    bool skipProcessingOnce = false;

    if(loadModel && modelPath != NULL)
    {
        tld->readFromFile(modelPath);
        _tld->readFromFile(modelPath);
        _tld1->readFromFile(modelPath);
        reuseFrameOnce = true;
    }
     else if(initialBB != NULL || (initialBB1 != NULL ) || (initialBB2 != NULL ))
    //  else if(initialBB != NULL )
    

    {
        Rect bb = tldArrayToRect(initialBB);
          Rect bb1 = tldArrayToRect(initialBB);
           Rect bb2 = tldArrayToRect(initialBB);


        printf("Starting at %d %d %d %d\n", bb.x, bb.y, bb.width, bb.height);
        printf("Starting at %d %d %d %d\n", bb1.x, bb1.y, bb1.width, bb1.height);
        printf("Starting at %d %d %d %d\n", bb2.x, bb2.y, bb2.width, bb2.height);


        tld->selectObject(grey, &bb);
        _tld->selectObject(grey, &bb);
         _tld1->selectObject(grey, &bb);


        skipProcessingOnce = true;
        reuseFrameOnce = true;
    }

    while(imAcqHasMoreFrames(imAcq))
    {
        double tic = cvGetTickCount();

        if(!reuseFrameOnce)
        {
            cvReleaseImage(&img);
            img = imAcqGetImg(imAcq);

            if(img == NULL)
            {
                printf("current image is NULL, assuming end of input.\n");
                break;
            }

            cvtColor(cvarrToMat(img), grey, CV_BGR2GRAY);
        }

        if(!skipProcessingOnce)
        {
            tld->processImage(cvarrToMat(img));
            _tld->processImage(cvarrToMat(img));
            _tld1->processImage(cvarrToMat(img));
            
        }
        else
        {
            skipProcessingOnce = false;
        }

        if(printResults != NULL)
        {
            if((tld->currBB) != NULL || (_tld ->currBB) || (_tld1 ->currBB) )
            {
                fprintf(resultsFile, "%d %.2d %.2d %.2d %.2d %f\n", imAcq->currentFrame - 1, tld->currBB->x, tld->currBB->y, tld->currBB->width, tld->currBB->height, tld->currConf);
                fprintf(resultsFile, "%d %.2d %.2d %.2d %.2d %f\n", imAcq->currentFrame - 1, _tld->currBB->x, _tld->currBB->y, _tld->currBB->width, _tld->currBB->height, _tld->currConf);
                fprintf(resultsFile, "%d %.2d %.2d %.2d %.2d %f\n", imAcq->currentFrame - 1, _tld1->currBB->x, _tld1->currBB->y, _tld1->currBB->width, _tld1->currBB->height, _tld1->currConf);
            }
            else
            {
                fprintf(resultsFile, "%d NaN NaN NaN NaN NaN\n", imAcq->currentFrame - 1);
            }
        }

        double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

        toc = toc / 1000000;

        float fps = 1 / toc;

         int confident = (tld->currConf >= threshold) ? 1 : 0;
        // // int confiden1 = (_tld->currConf >= threshold) ? 1 : 0;
        // // int confident2 = (_tld1->currConf >= threshold) ? 1 : 0;
        

        if(showOutput || saveDir != NULL)
        {
            char string[128];

            char learningString[10] = "";

            {
                strcpy(learningString, "Learning");
            }

            sprintf(string, "#%d,Posterior %.2f; fps: %.2f, #numwindows:%d, %s", imAcq->currentFrame - 1, tld->currConf, fps, tld->detectorCascade->numWindows, learningString);
            sprintf(string, "#%d,Posterior %.2f; fps: %.2f, #numwindows:%d, %s", imAcq->currentFrame - 1, _tld->currConf, fps, _tld->detectorCascade->numWindows, learningString);
            sprintf(string, "#%d,Posterior %.2f; fps: %.2f, #numwindows:%d, %s", imAcq->currentFrame - 1, _tld1->currConf, fps, _tld1->detectorCascade->numWindows, learningString);
            
            CvScalar yellow = CV_RGB(255, 255, 0);
            CvScalar blue = CV_RGB(0, 0, 255);
            CvScalar black = CV_RGB(0, 0, 0);
            CvScalar white = CV_RGB(255, 255, 255);
// .................
            if((_tld->currBB != NULL) || (tld->currBB != NULL ) || (_tld1->currBB != NULL ))
            {
                CvScalar rectangleColor = (confident) ? blue : yellow;

                cvRectangle(img, tld->currBB->tl(), tld->currBB->br(), rectangleColor, 8, 8, 0);
                cvRectangle(img, _tld->currBB->tl(), _tld->currBB->br(), rectangleColor, 8, 8, 0);
                cvRectangle(img, _tld1->currBB->tl(), _tld1->currBB->br(), rectangleColor, 8, 8, 0);
                

				if(showTrajectory)
				{
					CvPoint center = cvPoint(tld->currBB->x + tld->currBB->width/2, tld->currBB->y + tld->currBB->height/2);
					cvLine(img, cvPoint(center.x-2, center.y-2), cvPoint(center.x+2, center.y+2), rectangleColor, 2);
					cvLine(img, cvPoint(center.x-2, center.y+2), cvPoint(center.x+2, center.y-2), rectangleColor, 2);

					

                    CvPoint center1 = cvPoint(_tld->currBB->x + _tld->currBB->width/2, _tld->currBB->y + _tld->currBB->height/2);
					cvLine(img, cvPoint(center1.x-2, center1.y-2), cvPoint(center1.x+2, center1.y+2), rectangleColor, 2);
					cvLine(img, cvPoint(center1.x-2, center1.y+2), cvPoint(center1.x+2, center1.y-2), rectangleColor, 2);
					

                    CvPoint center11 = cvPoint(_tld1->currBB->x + _tld1->currBB->width/2, _tld1->currBB->y + _tld1->currBB->height/2);
					cvLine(img, cvPoint(center11.x-2, center1.y-2), cvPoint(center11.x+2, center11.y+2), rectangleColor, 2);
					cvLine(img, cvPoint(center11.x-2, center1.y+2), cvPoint(center11.x+2, center11.y-2), rectangleColor, 2);
					
					
					trajectory.addPoint(center, rectangleColor);
					trajectory.addPoint(center1, rectangleColor);
					trajectory.addPoint(center11, rectangleColor);

                    

				}
            }
			else if(showTrajectory)
			{
				trajectory.addPoint(cvPoint(-1, -1), cvScalar(-1, -1, -1));
			}

			if(showTrajectory)
			{
				trajectory.drawTrajectory(img);
			}

            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, 8);
            cvRectangle(img, cvPoint(0, 0), cvPoint(img->width, 50), black, CV_FILLED, 8, 0);
            cvPutText(img, string, cvPoint(25, 25), &font, white);

            cout <<" heloo world" << endl;


/// adjfhasdlkf  fix code

            if(showForeground)
            {

                for(size_t i = 0; i < tld->detectorCascade->detectionResult->fgList->size(); i++)
                {
                    Rect r = tld->detectorCascade->detectionResult->fgList->at(i);
                    cvRectangle(img, r.tl(), r.br(), white, 1);
                }

                for(size_t i = 0; i < _tld->detectorCascade->detectionResult->fgList->size(); i++)
                {
                    Rect r = _tld->detectorCascade->detectionResult->fgList->at(i);
                    cvRectangle(img, r.tl(), r.br(), white, 1);
                }

                for(size_t i = 0; i < _tld1->detectorCascade->detectionResult->fgList->size(); i++)
                {
                    Rect r = _tld1->detectorCascade->detectionResult->fgList->at(i);
                    cvRectangle(img, r.tl(), r.br(), white, 1);
                }

            }


            if(showOutput)
            {
                gui->showImage(img);
                char key = gui->getKey();

                if(key == 'q') break;

                if(key == 'b')
                {

                    ForegroundDetector *fg = tld->detectorCascade->foregroundDetector;

                    if(fg->bgImg.empty())
                    {
                        fg->bgImg = grey.clone();
                    }
                    else
                    {
                        fg->bgImg.release();
                    }
                }

                if(key == 'c')
                {
                    //clear everything
                    tld->release();
                }

                if(key == 'l')
                {
                    tld->learningEnabled = !tld->learningEnabled;
                    printf("LearningEnabled: %d\n", tld->learningEnabled);
                }

                if(key == 'r')
                {
                    CvRect box;
                    CvRect _box;
                    CvRect _box1;
                    
                    getBBFromUser(img, box, gui);
                    getBBFromUser(img, _box, gui);
                    

                    if(getBBFromUser(img, _box1, gui) == PROGRAM_EXIT)
                    {
                        break;
                    }

                    Rect r = Rect(box);
                    Rect _r = Rect(_box);
                    Rect _r1 = Rect(_box1);

                    tld ->selectObject(grey, &r);
                    _tld ->selectObject(grey, &_r);
                    _tld1->selectObject(grey, &_r1);
                }
        
            }

            if(saveDir != NULL)
            {
                char fileName[256];
                sprintf(fileName, "%s/%.5d.png", saveDir, imAcq->currentFrame - 1);

                cvSaveImage(fileName, img);
            }
        }

        if(reuseFrameOnce)
        {
            reuseFrameOnce = false;
        }
    }

    cvReleaseImage(&img);
    img = NULL;

    if(exportModelAfterRun)
    {
        tld->writeToFile(modelExportFile);
        _tld->writeToFile(modelExportFile);
        _tld1->writeToFile(modelExportFile);
    }

    if(resultsFile)
    {
        fclose(resultsFile);
    }
}
