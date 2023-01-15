#include "Matrix.h"
#include <cstdlib>

using namespace std;

inline float Sigmond(float x)
{
	return 1.0 / (1 + exp(-x));
}

inline float DSigmond(float x)
{
	return x * (1 - x);
}
class NeuralNet
{
public:
	vector<uint32_t> _topology;
	vector<Matrix> _weightMatrices; //
	vector<Matrix> _valueMatrices;
	vector<Matrix> _biasMatrices; //same dimensions as value Matrices
	float _learningRate;

public:
	NeuralNet(vector<uint32_t> topology, float learningRate = 0.1f) : _topology(topology), _weightMatrices({}), _valueMatrices({}), _biasMatrices({}), _learningRate(learningRate)
	{
		for (uint32_t i = 0; i < topology.size() - 1; i++)
		{
			Matrix weightMatrix(topology[i + 1], topology[i]);
			weightMatrix = weightMatrix.applyFunction([](const float& f) {
				return (float)rand() / RAND_MAX;
				});
			_weightMatrices.push_back(weightMatrix);

			Matrix biasMatrix(topology[i + 1], 1);
			biasMatrix = biasMatrix.applyFunction([](const float& f) {
				return (float)rand() / RAND_MAX;
				});
			_biasMatrices.push_back(biasMatrix);

			_valueMatrices.resize(topology.size());
		}
	} 

	bool feedForward(vector<float> input)
	{
		if (input.size() != _topology[0])
			return false;
		Matrix values(input.size(), 1);
		for (uint32_t i = 0; i < input.size(); i++)
			values._vals[i] = input[i];

		//feed forward to next layers 
		for (uint32_t i = 0; i < _weightMatrices.size(); i++)
		{
			_valueMatrices[i] = values;
			values = values.multiply(_weightMatrices[i]);
			values = values.add(_biasMatrices[i]);
			values = values.applyFunction(Sigmond);
		}

		_valueMatrices[_weightMatrices.size()] = values;

		return true;
	}

	bool backPropagate(vector<float> targetOutput)
	{
		if (targetOutput.size() != _topology.back())
			return false;
		Matrix errors(targetOutput.size(), 1);
		Matrix sub = _valueMatrices.back().negative();
		errors._vals = targetOutput;
		errors = errors.add(sub);

		for (int i = _weightMatrices.size() - 1; i >= 0; i--)
		{
			Matrix trans = _weightMatrices[i].transpose();
			Matrix prevErrors = errors.multiply(trans);
			Matrix dOuptputs = _valueMatrices[i + 1].applyFunction(DSigmond);
			Matrix gradients = errors.multiplyElements(dOuptputs);
			gradients = gradients.multiplyScalar(_learningRate);
			Matrix weightGradients = _valueMatrices[i].transpose().multiply(gradients);

			_weightMatrices[i] = _weightMatrices[i].add(weightGradients);
			_biasMatrices[i] = _biasMatrices[i].add(gradients);
			errors = prevErrors;
		}

		return  true;
	}

	vector<float> getPrediction()
	{
		return _valueMatrices.back()._vals;
	}
};

