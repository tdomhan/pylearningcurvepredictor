import unittest
import numpy as np
import os
import argparse

from subprocess import Popen, PIPE

from caffe.proto import caffe_pb2

from pylrpredictor.curvefunctions import  all_models, model_defaults
from pylrpredictor.terminationcriterion import main


def write_xlim(xlim, test_interval=2):
    solver = caffe_pb2.SolverParameter()

    solver.max_iter = xlim * test_interval
    solver.test_interval = test_interval

    open("caffenet_solver.prototxt", "w").write(str(solver))


def run_program(cmds):
    process = Popen(cmds, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    return exit_code


real_abort_learning_curve = [ 0.1126,  0.3304,  0.3844,  0.3984,  0.4128,  0.4366,  0.4536,
        0.4664,  0.4742,  0.4858,  0.495 ,  0.5038,  0.5136,  0.5198,
        0.522 ,  0.5298,  0.539 ,  0.5422,  0.5452,  0.5508,  0.5556,
        0.5586,  0.5624,  0.5668,  0.5722,  0.5752,  0.5774,  0.579 ,
        0.5822,  0.5848,  0.5852,  0.5878,  0.5904,  0.594 ,  0.5974,
        0.5992,  0.5988,  0.6006,  0.6022,  0.6042,  0.6054,  0.6058,
        0.6072,  0.6106,  0.6152,  0.6112,  0.6142,  0.6152,  0.6148,
        0.6172,  0.6168,  0.6194,  0.6198,  0.6208,  0.6206,  0.626 ,
        0.627 ,  0.627 ,  0.626 ,  0.6272,  0.6268,  0.6256,  0.6314,
        0.6318,  0.6318,  0.6368,  0.6346,  0.6354,  0.6376,  0.6356,
        0.637 ,  0.6394,  0.6426,  0.6432,  0.6418,  0.6428,  0.6448,
        0.6436,  0.6456,  0.6454,  0.649 ,  0.648 ,  0.6494,  0.6492,
        0.6504,  0.6492,  0.651 ,  0.6502,  0.653 ,  0.653 ,  0.6518,
        0.6564,  0.656 ,  0.6552,  0.6542,  0.656 ,  0.655 ,  0.6564,
        0.657 ,  0.6572,  0.6612,  0.6624,  0.6636,  0.6604,  0.6604,
        0.662 ,  0.6604,  0.66  ,  0.6576,  0.6614,  0.6644,  0.6658,
        0.6676,  0.6688,  0.6676,  0.6686,  0.6678,  0.6652,  0.666 ,
        0.67  ,  0.6674,  0.6708,  0.6714,  0.6708,  0.6724,  0.671 ,
        0.6702,  0.6716,  0.6716,  0.6736,  0.6692,  0.6742,  0.6766,
        0.6768,  0.6772,  0.676 ,  0.6772,  0.6772,  0.6788,  0.678 ,
        0.6806,  0.6784,  0.682 ,  0.6822,  0.6822,  0.6816,  0.6834,
        0.6822,  0.6828,  0.683 ,  0.6858,  0.6838,  0.6826,  0.6886,
        0.6882,  0.6866,  0.6882,  0.6914,  0.6894,  0.6876,  0.685 ,
        0.6902,  0.6876,  0.6936,  0.694 ,  0.6948,  0.6922,  0.6936,
        0.695 ,  0.691 ,  0.6886,  0.6896,  0.6942,  0.6918,  0.6962,
        0.698 ,  0.699 ,  0.6964,  0.6994,  0.698 ,  0.6952,  0.6932,
        0.6958,  0.6958,  0.698 ,  0.7024,  0.7028,  0.6992,  0.7006,
        0.7038,  0.7016,  0.6986,  0.6994,  0.7012,  0.7   ,  0.7046,
        0.704 ,  0.703 ,  0.7038,  0.701 ,  0.7046,  0.7036,  0.7026,
        0.7   ,  0.705 ,  0.7034,  0.7084,  0.7084,  0.7068,  0.7078,
        0.7098,  0.7078,  0.7076,  0.705 ,  0.705 ,  0.7074,  0.7084,
        0.711 ,  0.7054,  0.7102,  0.7118,  0.7104,  0.7088,  0.7088,
        0.7104,  0.7112,  0.7094,  0.714 ,  0.7136,  0.7138,  0.716 ,
        0.7146,  0.713 ,  0.711 ,  0.7108,  0.7124,  0.714 ,  0.712 ,
        0.7166,  0.7152,  0.713 ,  0.7178,  0.716 ,  0.7122,  0.715 ,
        0.7154,  0.7128,  0.7156,  0.7162,  0.7178,  0.7176,  0.7202,
        0.7212,  0.7164,  0.7164,  0.718 ,  0.7172,  0.7188,  0.718 ,
        0.7204,  0.719 ,  0.721 ,  0.7222,  0.7216,  0.7198,  0.719 ,
        0.7214,  0.7196,  0.7206,  0.7216,  0.7236,  0.723 ,  0.724 ,
        0.7234,  0.7236,  0.7238,  0.7208,  0.7202,  0.7198,  0.7226,
        0.7228,  0.7236,  0.7262,  0.7244,  0.7218,  0.7204,  0.7238,
        0.7232,  0.724 ,  0.7244,  0.727 ,  0.7266,  0.7278,  0.7262,
        0.7274,  0.7246,  0.724 ,  0.725 ,  0.7254,  0.7236,  0.726 ,
        0.7244,  0.7272,  0.7294,  0.7274,  0.7284,  0.7254,  0.725 ,
        0.7242,  0.7278,  0.7272,  0.726 ,  0.7274,  0.7272,  0.73  ,
        0.7302,  0.7286,  0.7238,  0.7294,  0.7286,  0.7264,  0.73  ,
        0.7274,  0.7326,  0.7286,  0.7304,  0.7322,  0.7274,  0.7258,
        0.7296,  0.7268,  0.7262,  0.7282,  0.7294,  0.7336,  0.7338,
        0.7328,  0.7316,  0.7286,  0.7322,  0.7318,  0.732 ,  0.7302,
        0.732 ,  0.734 ,  0.7314,  0.7356,  0.7352,  0.7302,  0.7284,
        0.732 ,  0.732 ,  0.7298,  0.733 ,  0.735 ,  0.7342,  0.7312,
        0.7346,  0.7358,  0.7318,  0.732 ,  0.733 ,  0.735 ,  0.7318,
        0.735 ,  0.7334,  0.7348,  0.7366,  0.7356,  0.734 ,  0.7336,
        0.7334,  0.7324,  0.734 ,  0.7344,  0.7348,  0.736 ,  0.7346,
        0.7342,  0.7374,  0.7362,  0.732 ,  0.7324,  0.7368,  0.7346,
        0.7334,  0.7356,  0.7374,  0.7372,  0.7354,  0.7364,  0.7338,
        0.735 ,  0.733 ,  0.7354,  0.7326,  0.7364,  0.7372,  0.7372,
        0.7364,  0.7356,  0.7384,  0.7344,  0.734 ,  0.7326,  0.7378,
        0.7348,  0.7376,  0.7374,  0.737 ,  0.7394,  0.739 ,  0.7372,
        0.7366,  0.7378,  0.7366,  0.736 ,  0.7356,  0.7346,  0.7388,
        0.7348,  0.7378,  0.7388,  0.7378,  0.7356,  0.7354,  0.738 ,
        0.7376,  0.7396,  0.7402,  0.741 ,  0.7366,  0.7382,  0.7422,
        0.7414,  0.7364,  0.736 ,  0.739 ,  0.7358,  0.738 ,  0.7396,
        0.74  ,  0.74  ,  0.7432,  0.7416,  0.7384,  0.7404,  0.7378,
        0.737 ,  0.7384,  0.741 ,  0.7448,  0.7408,  0.741 ,  0.7458,
        0.7412,  0.7384,  0.7408,  0.74  ,  0.737 ,  0.7404,  0.7416,
        0.7414,  0.7396,  0.7408,  0.7446,  0.7432,  0.7416,  0.7376,
        0.7402,  0.7364,  0.7404,  0.7418,  0.7408,  0.7422,  0.7426,
        0.7408,  0.741 ,  0.7426,  0.7368,  0.7392,  0.739 ,  0.7412,
        0.742 ,  0.737 ,  0.7426,  0.746 ,  0.7394,  0.7392,  0.743 ,
        0.742 ,  0.7372,  0.7404,  0.741 ,  0.7436,  0.74  ,  0.7398,
        0.7472,  0.742 ,  0.744 ,  0.742 ,  0.7452,  0.7382,  0.7406,
        0.7414,  0.7406,  0.7398,  0.7452,  0.7436,  0.7414,  0.7444,
        0.7412,  0.7436,  0.741 ,  0.74  ,  0.7438,  0.7414,  0.745 ,
        0.7462,  0.7446,  0.741 ,  0.7438,  0.7428,  0.7422,  0.7412,
        0.741 ,  0.7452,  0.7428,  0.7462,  0.7464,  0.7454,  0.7436,
        0.741 ,  0.745 ,  0.7388,  0.7422,  0.746 ,  0.7426,  0.7428,
        0.7466,  0.7464,  0.7452,  0.744 ,  0.7456,  0.742 ,  0.7394,
        0.741 ,  0.7448,  0.7456,  0.742 ,  0.7458,  0.7444,  0.7446,
        0.745 ,  0.743 ,  0.743 ,  0.7432,  0.7432,  0.742 ,  0.7452,
        0.7468,  0.745 ,  0.7452,  0.7438,  0.742 ,  0.7436,  0.7444,
        0.7428,  0.7452,  0.7452,  0.7462,  0.747 ,  0.7492,  0.7454,
        0.7454,  0.7454,  0.7462,  0.742 ,  0.7446,  0.7466,  0.7476,
        0.7474,  0.747 ,  0.7454,  0.7412,  0.747 ,  0.7462,  0.7474,
        0.7452,  0.7454,  0.7474,  0.7474,  0.7478,  0.7466,  0.7464,
        0.7456]
real_abort_ybest = .80766
real_abort_xlim = 2850


class Terminationcriterion(unittest.TestCase):

    def test_conservative_predict_cancel(self):
        """
            The termination criterion expects the learning_curve in a file
            called learning_curve.txt as well as the current best value in 
            ybest.txt. We create both files and see if the termination criterion
            correctly predicts to cancel or continue running under various artificial
            ybest.
        """
        for prob_x_greater_type in ["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"]:
            np.random.seed(13)
            #generate some data:
            for model_name in ["pow3", "log_power"]:
                function = all_models[model_name]
                params = model_defaults[model_name]
                xlim = 500
                x = np.arange(1, xlim, 1)
                y = function(x, **params)
                noise = 0.0005 * np.random.randn(len(y))
                y_noisy = y + noise
                y_final = y_noisy[-1]
                num_train = 200
                np.savetxt("learning_curve.txt", y_noisy[:200])
                write_xlim(xlim)

                print "Actual ybest: %f" % y_noisy[-1]

                #we set ybest to be higher than the final value of this curve
                #hence we DO want the evaluation to stop!
                open("ybest.txt", "w").write(str(y_final + 0.05))
                open("termination_criterion_running", "w").write("running")

                ret = main(mode="conservative",
                    prob_x_greater_type=prob_x_greater_type,
                    nthreads=4)
                self.assertEqual(ret, 1)

                self.assertTrue(os.path.exists("y_predict.txt"))
                y_predict = float(open("y_predict.txt").read())
                abserr = np.abs(y_predict - y_noisy[-1])
                print "abs error %f" % abserr
                self.assertTrue(abserr < 0.03)

                #we set ybest to be lower than the final value of this curve
                #hence we DON'T want the evaluation to stop!
                open("ybest.txt", "w").write(str(y_final - 0.05))
                open("termination_criterion_running", "w").write("running")

                ret = main(mode="conservative",
                    prob_x_greater_type=prob_x_greater_type,
                    nthreads=4)
                self.assertEqual(ret, 0)
                self.assertFalse(os.path.exists("y_predict.txt"))
                self.assertFalse(os.path.exists("termination_criterion_running"))
                self.assertFalse(os.path.exists("termination_criterion_running_pid"))
        self.cleanup()

    def test_conservative_real_example(self):
        """
            The termination criterion expects the learning_curve in a file
            called learning_curve.txt as well as the current best value in 
            ybest.txt. We create both files and see if the termination criterion
            correctly predicts to cancel or continue running under various artificial
            ybest.
        """
        for prob_x_greater_type in ["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"]:
            np.savetxt("learning_curve.txt", real_abort_learning_curve)
            write_xlim(real_abort_xlim)

            open("ybest.txt", "w").write(str(real_abort_ybest))
            open("termination_criterion_running", "w").write("running")

            ret = main(mode="conservative",
                prob_x_greater_type=prob_x_greater_type,
                nthreads=4)
            #ybest is higher than what the curve will ever reach
            #hence we expect to cancel the run:
            self.assertEqual(ret, 1)

            self.assertTrue(os.path.exists("y_predict.txt"))
            self.assertFalse(os.path.exists("termination_criterion_running"))
            self.assertFalse(os.path.exists("termination_criterion_running_pid"))
        self.cleanup()

    def test_conservative_command_line_args(self):
        for prob_x_greater_type in ["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"]:
            np.random.seed(13)
            #generate some data:
            for model_name in ["pow3", "log_power"]:
                function = all_models[model_name]
                params = model_defaults[model_name]
                xlim = 500
                x = np.arange(1, xlim, 1)
                y = function(x, **params)
                noise = 0.0005 * np.random.randn(len(y))
                y_noisy = y + noise
                y_final = y_noisy[-1]
                num_train = 200
                np.savetxt("learning_curve.txt", y_noisy[:200])
                write_xlim(xlim)

                print "Actual ybest: %f" % y_noisy[-1]

                #we set ybest to be higher than the final value of this curve
                #hence we DO want the evaluation to stop!
                open("ybest.txt", "w").write(str(y_final + 0.05))
                open("termination_criterion_running", "w").write("running")

                ret = run_program(["python", "-m", "pylrpredictor.terminationcriterion",
                    "--nthreads", "5",
                    "--mode", "conservative", 
                    "--prob-x-greater-type", prob_x_greater_type])
                self.assertEqual(ret, 1)

                self.assertTrue(os.path.exists("y_predict.txt"))
                y_predict = float(open("y_predict.txt").read())

                #we set ybest to be lower than the final value of this curve
                #hence we DON'T want the evaluation to stop!
                open("ybest.txt", "w").write(str(y_final - 0.05))
                open("termination_criterion_running", "w").write("running")

                ret = run_program(["python", "-m", "pylrpredictor.terminationcriterion",
                    "--nthreads", "5",
                    "--mode", "conservative", 
                    "--prob-x-greater-type", prob_x_greater_type])
                self.assertEqual(ret, 0)
                self.assertFalse(os.path.exists("y_predict.txt"))
                self.assertFalse(os.path.exists("termination_criterion_running"))
                self.assertFalse(os.path.exists("termination_criterion_running_pid"))

        self.cleanup()


    def test_conservative_real_example_command_line_args(self):
        """
            The termination criterion expects the learning_curve in a file
            called learning_curve.txt as well as the current best value in 
            ybest.txt. We create both files and see if the termination criterion
            correctly predicts to cancel or continue running under various artificial
            ybest.
        """
        for prob_x_greater_type in ["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"]:
            np.savetxt("learning_curve.txt", real_abort_learning_curve)
            write_xlim(real_abort_xlim)

            open("ybest.txt", "w").write(str(real_abort_ybest))
            open("termination_criterion_running", "w").write("running")

            ret = run_program(["python", "-m", "pylrpredictor.terminationcriterion",
                    "--nthreads", "5",
                    "--mode", "conservative", 
                    "--prob-x-greater-type", prob_x_greater_type,
                    #just check that it accepts the value for predictive-std-threshold, it's set too high to have a real influenec
                    "--predictive-std-threshold", "10."])
            #ybest is higher than what the curve will ever reach
            #hence we expect to cancel the run:
            self.assertEqual(ret, 1)

            self.assertTrue(os.path.exists("y_predict.txt"))
            self.assertFalse(os.path.exists("termination_criterion_running"))
            self.assertFalse(os.path.exists("termination_criterion_running_pid"))
        self.cleanup()


    def test_optimistic_real_example_command_line_args(self):
        """
            The termination criterion expects the learning_curve in a file
            called learning_curve.txt as well as the current best value in 
            ybest.txt. We create both files and see if the termination criterion
            correctly predicts to cancel or continue running under various artificial
            ybest.
        """
        for prob_x_greater_type in ["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"]:
            np.savetxt("learning_curve.txt", real_abort_learning_curve)
            write_xlim(real_abort_xlim)

            open("ybest.txt", "w").write(str(real_abort_ybest))
            open("termination_criterion_running", "w").write("running")

            ret = run_program(["python", "-m", "pylrpredictor.terminationcriterion",
                    "--nthreads", "5",
                    "--mode", "optimistic", 
                    "--predictive-std-threshold", str(0.05)])
            #ybest is higher than what the curve will ever reach
            #hence we expect to cancel the run:
            self.assertEqual(ret, 1)

            ret = run_program(["python", "-m", "pylrpredictor.terminationcriterion",
                    "--nthreads", "5",
                    "--mode", "optimistic", 
                    "--predictive-std-threshold", str(0.01)])
            #ybest is higher than what the curve will ever reach
            #hence we expect to cancel the run:
            self.assertEqual(ret, 1)

            ret = run_program(["python", "-m", "pylrpredictor.terminationcriterion",
                    "--nthreads", "5",
                    "--mode", "optimistic"])
            #ybest is higher than what the curve will ever reach
            #hence we expect to cancel the run:
            self.assertEqual(ret, 1)

            self.assertTrue(os.path.exists("y_predict.txt"))
            self.assertFalse(os.path.exists("termination_criterion_running"))
            self.assertFalse(os.path.exists("termination_criterion_running_pid"))
        self.cleanup()



    def test_conservative_predictive_std_predict_cancel(self):
        for prob_x_greater_type in ["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"]:
            np.random.seed(13)
            #generate some data:

            model_name = "pow3"
            function = all_models[model_name]

            params = {'a': 0.52, 'alpha': 0.2, 'c': 0.84}
            xlim = 500
            x = np.arange(1, xlim, 1)
            y = function(x, **params)
            noise = 0.01 * np.random.randn(len(y))
            y_noisy = y + noise
            y_final = y_noisy[-1]
            num_train = 30
            np.savetxt("learning_curve.txt", y_noisy[:num_train])
            write_xlim(xlim)

            #first check:
            #if there's no ybest and the predictive_std is high
            #then we want the evaluation to continue
            if os.path.exists("ybest.txt"):
                os.remove("ybest.txt")
            ret = main(mode="conservative",
                prob_x_greater_type=prob_x_greater_type,
                predictive_std_threshold=0.00001,
                nthreads=4)
            self.assertEqual(ret, 0)

            print "Actual ybest: %f" % y_noisy[-1]

            #we set ybest to be higher than the final value of this curve
            #BUT because the predictive std is still high we don't want to stop
            open("ybest.txt", "w").write(str(y_final + 0.05))
            open("termination_criterion_running", "w").write("running")

            ret = main(mode="conservative",
                prob_x_greater_type=prob_x_greater_type,
                predictive_std_threshold=0.00001,
                nthreads=4)
            self.assertEqual(ret, 0)

            self.assertFalse(os.path.exists("y_predict.txt"))

        self.cleanup()


    def test_optimistic_predict_cancel(self):
        """
            Optimisitic mode

            The termination criterion expects the learning_curve in a file
            called learning_curve.txt as well as the current best value in 
            ybest.txt. We create both files and see if the termination criterion
            correctly predicts to cancel or continue running under various artificial
            ybest.
        """
        for prob_x_greater_type in ["posterior_mean_prob_x_greater_than", "posterior_prob_x_greater_than"]:
            np.random.seed(13)
            #generate some data:

            model_name = "pow3"
            function = all_models[model_name]

            params = {'a': 0.52, 'alpha': 0.2, 'c': 0.84}
            xlim = 500
            x = np.arange(1, xlim, 1)
            y = function(x, **params)
            noise = 0.01 * np.random.randn(len(y))
            y_noisy = y + noise
            y_final = y_noisy[-1]
            num_train = 30
            np.savetxt("learning_curve.txt", y_noisy[:num_train])
            write_xlim(xlim)

            #first check:
            #if there's no ybest and the predictive_std is high
            #then we want the evaluation to continue
            if os.path.exists("ybest.txt"):
                os.remove("ybest.txt")
            ret = main(mode="optimistic",
                prob_x_greater_type=prob_x_greater_type,
                nthreads=4)
            self.assertEqual(ret, 0)

            print "Actual ybest: %f" % y_noisy[-1]

            #we set ybest to be higher than the final value of this curve
            #hence we DO want the evaluation to stop!
            open("ybest.txt", "w").write(str(y_final + 0.05))
            open("termination_criterion_running", "w").write("running")

            ret = main(mode="optimistic",
                prob_x_greater_type=prob_x_greater_type,
                nthreads=4)
            self.assertEqual(ret, 1)

            self.assertTrue(os.path.exists("y_predict.txt"))
            y_predict = float(open("y_predict.txt").read())
            abserr = np.abs(y_predict-y_noisy[-1])
            self.assertTrue(abserr < 0.05)
            print "abs error %f" % abserr

            #we set ybest to be lower than the final value of this curve
            #hence we DON'T want the evaluation to stop!
            #we assume here that because the model was set up like this 
            #the predictive_std is above (it should actually be around 0.019)
            open("ybest.txt", "w").write(str(y_final - 0.05))
            open("termination_criterion_running", "w").write("running")

            ret = main(mode="optimistic", nthreads=4)
            self.assertEqual(ret, 0)
            self.assertFalse(os.path.exists("y_predict.txt"))
            self.assertFalse(os.path.exists("termination_criterion_running"))
            self.assertFalse(os.path.exists("termination_criterion_running_pid"))

            num_train = 300
            np.savetxt("learning_curve.txt", y_noisy[:num_train])
            #we set ybest to be lower than the final value of this curve
            #HOWEVER we except the predictive std to be around .0027
            #so the the run should be cancelled nevertheless
            open("ybest.txt", "w").write(str(y_final - 0.05))
            open("termination_criterion_running", "w").write("running")

            ret = main(mode="optimistic",
                prob_x_greater_type=prob_x_greater_type,
                nthreads=4)
            self.assertEqual(ret, 1)
            self.assertTrue(os.path.exists("y_predict.txt"))
            y_predict = float(open("y_predict.txt").read())
            abserr = np.abs(y_predict-y_noisy[-1])
            self.assertTrue(abserr < 0.05)
            print "abs error %f" % abserr

            self.assertFalse(os.path.exists("termination_criterion_running"))
            self.assertFalse(os.path.exists("termination_criterion_running_pid"))

        self.cleanup()


    def test_error_logging(self):
        """
            Test in case of an error, the error will be logged.
        """
        open("ybest.txt", "w").write(str(0.5))
        #Let's e.g. run main without creating any files
        if os.path.exists("learning_curve.txt"):
            os.remove("learning_curve.txt")
        ret = main()
        self.assertTrue(os.path.exists("term_crit_error.txt"))

        os.remove("ybest.txt")


    def cleanup(self):
        if os.path.exists("learning_curve.txt"):
            os.remove("learning_curve.txt")
        if os.path.exists("ybest.txt"):    
            os.remove("ybest.txt")
        if os.path.exists("termination_criterion_running"):    
            os.remove("termination_criterion_running")
        if os.path.exists("term_crit_error.txt"):    
            os.remove("term_crit_error.txt")


    def test_predict_no_cancel(self):
        pass



if __name__ == "__main__":
    unittest.main()
