#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

#include "network.h"
#include "tools.h"

void NeuralNetwork::add(ILayer *layer)
{
    this->nn.push_back(layer);
    return;
}

void NeuralNetwork::miniBatch(xt::xarray<float> batch , std::vector<bool> trueLabels, uint16_t dropRate)
{
    this->dropDense(dropRate);
    for (int i = 1; i < batch.shape()[0]; ++i)
    {   
        this->train(xt::view(batch, i) , trueLabels[i]) ;
    }
}

void NeuralNetwork::dropDense(uint16_t dropRate){

    for(int i = 0 ; i < this->nn.size() ; ++i)
    {
        if (Dense *dense = dynamic_cast<Dense*>(this->nn[i]))   // le sheitan
        {
            dense->dropout(dropRate);
        }
    }
}

void NeuralNetwork::train(xt::xarray<float> input, bool trueLabel )
{


    this->nn[0]->forward(input);

    for(int i = 1 ; i < this->nn.size() ; ++i)
    {
        this->nn[i]->forward(this->nn[i-1]->output);
    }

    
    auto error = lossFunction(this->nn[this->nn.size()-1]->output , trueLabel);

    std::cout << "output: " << this->nn[this->nn.size()-1]->output << '\n'
              << "error: " << error;

    for(int i = 0 ; i < this->nn.size() ; ++i)
    {
        if (Dense *dense = dynamic_cast<Dense*>(this->nn[i]))
        {   
            // learning rate = 0.01
            dense->backward(error,0.01);
        }
    }
}

void NeuralNetwork::detect( xt::xarray<float> input)
{}

void NeuralNetwork::load(const char * path)
{}

void NeuralNetwork::save(const char * path) const
{}