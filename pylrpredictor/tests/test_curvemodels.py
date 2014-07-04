from pylrpredictor.curvemodels import CurveModel, MLCurveMovel, LinearCurveModel
import unittest
import numpy as np
import random

from pylrpredictor.curvefunctions import  all_models, model_defaults

class CurveModelTest(unittest.TestCase):

	def test_interface(self):
		model = CurveModel(function=lambda x: x)
		x = np.arange(0, 10)
		y = np.arange(0, 10)
		self.assertRaises(NotImplementedError, model.fit, x, y)
		#self.assertRaises(NotImplementedError, model.posterior_log_likelihood, x, y)

	def test_ml_fit(self):
	    for model_name in all_models.keys():
	        if model_name == "linear":
	            m = LinearCurveModel()
	        else:
	            if model_name in model_defaults:
	                m = MLCurveMovel(function=all_models[model_name],
	                                 default_vals=model_defaults[model_name],
	                                 recency_weighting=True)
	            else:
	                m = MLCurveMovel(function=all_models[model_name], recency_weighting=True)

	        #generate some data for the model
	        x = np.arange(1, 1000)
	        params = m.default_function_param_array()
	        params =  params + np.random.rand(params.shape[0])
	        y = m.function(x, *params)
	        std = 0.01
	        y += std*np.random.randn(y.shape[0])
	        self.assertTrue(m.fit(x, y))
	        print "original params vs fit params:"
	        print params
	        print m.ml_params

	#def test_ml_fit(self):
#		model1 = random.choice(all_models.keys())
#		model1 = random.choice(all_models.keys())

if __name__ == "__main__":
	unittest.main()