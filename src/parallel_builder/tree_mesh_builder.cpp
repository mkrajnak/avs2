/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  MARTIN KRAJNAK <xkrajn02@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    24.11.2019
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::evaluateCube(const Vec3_t<float> &pos, size_t gridSize, const ParametricScalarField &field)
{   
    size_t totalCubesCount = gridSize*4;
    size_t step = gridSize/2;
    unsigned totalTrianglesCount1 = 0, totalTrianglesCount2 = 0;

    for(size_t i = 0; i < totalCubesCount; i+=step)
    {
        // 3. Compute 3D position in the grid.
        #pragma omp task firstprivate(step) shared(totalTrianglesCount1, totalTrianglesCount2)
        {
            Vec3_t<float> cubeOffset(pos.x + (i >= (gridSize*2) ? step : 0),
                                     pos.y + ((i/gridSize) % 2 ? step : 0),
                                     pos.z + (i % gridSize));
            
            Vec3_t<float> middleOffset((cubeOffset.x + step/2)*mGridResolution,
                                       (cubeOffset.y + step/2)*mGridResolution,
                                       (cubeOffset.z + step/2)*mGridResolution);

            if (step == 1)
            {
                totalTrianglesCount1 += buildCube(cubeOffset, field);
            }
            else if ((evaluateFieldAt(middleOffset, field) < ((sqrt(3)*step*mGridResolution)/2)+mIsoLevel))
            {
                totalTrianglesCount2 += evaluateCube(cubeOffset, step, field);                
            }
        }
    }
    #pragma omp taskwait 
    return totalTrianglesCount1 + totalTrianglesCount2;
}


unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    Vec3_t<float> startOffset(0,0,0);
    unsigned triangles = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {  
            triangles = evaluateCube(startOffset, mGridSize, field);
        }
    }
    return triangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    #pragma omp simd reduction (min:value)
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    #pragma omp critical (c1)
    mTriangles.push_back(triangle);
}
