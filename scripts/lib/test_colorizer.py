import unittest
import numpy as np
import numpy.testing as npt
import colorizer as C

class Test_create_convert_linear(unittest.TestCase):
    
    def test_create_convert_linear1(self):
        fn = C.create_convert_linear(0.0, 1.0)
        
        # in
        a1 = np.array([ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
        e1 = np.array([-1.0, -0.75, -0.5, -0.25,  0.0, 0.25,  0.5,  0.75,  1.0])
        f1 = fn(a1)
        npt.assert_equal(f1, e1)
        
        # lo-in
        a2 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0,  0.25, 0.5, 0.75, 1.0])
        e2 = np.array([-1.0, -1.0,  -1.0, -1.0, -1.0, -0.5,  0.0, 0.5,  1.0])
        f2 = fn(a2)
        npt.assert_equal(f2, e2)
        
        # in-hi
        a3 = np.array([ 0.0,  0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        e3 = np.array([-1.0, -0.5,  0.0, 0.5,  1.0, 1.0,  1.0,  1.0, 1.0])
        f3 = fn(a3)
        npt.assert_equal(f3, e3)
        
        # lo-hi
        a4 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0,  0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        e4 = np.array([-1.0, -1.0,  -1.0, -1.0, -1.0, -0.5,  0.0, 0.5,  1.0, 1.0,  1.0, 1.0,  1.0])
        f4 = fn(a4)
        npt.assert_equal(f4, e4)
    
    def test_create_convert_linear2(self):
        fn = C.create_convert_linear(1.0, 2.0)
        
        # in
        a1 = np.array([ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]) + 1.0
        e1 = np.array([-1.0, -0.75, -0.5, -0.25,  0.0, 0.25,  0.5,  0.75,  1.0])
        f1 = fn(a1)
        npt.assert_equal(f1, e1)
        
        # lo-in
        a2 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0,  0.25, 0.5, 0.75, 1.0]) + 1.0
        e2 = np.array([-1.0, -1.0,  -1.0, -1.0, -1.0, -0.5,  0.0, 0.5,  1.0])
        f2 = fn(a2)
        npt.assert_equal(f2, e2)
        
        # in-hi
        a3 = np.array([ 0.0,  0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) + 1.0
        e3 = np.array([-1.0, -0.5,  0.0, 0.5,  1.0, 1.0,  1.0,  1.0, 1.0])
        f3 = fn(a3)
        npt.assert_equal(f3, e3)
        
        # lo-hi
        a4 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0,  0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) + 1.0
        e4 = np.array([-1.0, -1.0,  -1.0, -1.0, -1.0, -0.5,  0.0, 0.5,  1.0, 1.0,  1.0, 1.0,  1.0])
        f4 = fn(a4)
        npt.assert_equal(f4, e4)
    
    def test_create_convert_linear3(self):
        fn = C.create_convert_linear(-1.0, 1.0)
        
        # in
        a1 = np.array([ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]) * 2.0 - 1.0
        e1 = np.array([-1.0, -0.75, -0.5, -0.25,  0.0, 0.25,  0.5,  0.75,  1.0])
        f1 = fn(a1)
        npt.assert_equal(f1, e1)
        
        # lo-in
        a2 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0,  0.25, 0.5, 0.75, 1.0]) * 2.0 - 1.0
        e2 = np.array([-1.0, -1.0,  -1.0, -1.0, -1.0, -0.5,  0.0, 0.5,  1.0])
        f2 = fn(a2)
        npt.assert_equal(f2, e2)
        
        # in-hi
        a3 = np.array([ 0.0,  0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) * 2.0 - 1.0
        e3 = np.array([-1.0, -0.5,  0.0, 0.5,  1.0, 1.0,  1.0,  1.0, 1.0])
        f3 = fn(a3)
        npt.assert_equal(f3, e3)
        
        # lo-hi
        a4 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0,  0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) * 2.0 - 1.0
        e4 = np.array([-1.0, -1.0,  -1.0, -1.0, -1.0, -0.5,  0.0, 0.5,  1.0, 1.0,  1.0, 1.0,  1.0])
        f4 = fn(a4)
        npt.assert_equal(f4, e4)
    
    def test_create_convert_linear4(self):
        fn = C.create_convert_linear(-2.0, -1.0)
        
        # in
        a1 = np.array([ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]) - 2.0
        e1 = np.array([-1.0, -0.75, -0.5, -0.25,  0.0, 0.25,  0.5,  0.75,  1.0])
        f1 = fn(a1)
        npt.assert_equal(f1, e1)
        
        # lo-in
        a2 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0,  0.25, 0.5, 0.75, 1.0]) - 2.0
        e2 = np.array([-1.0, -1.0,  -1.0, -1.0, -1.0, -0.5,  0.0, 0.5,  1.0])
        f2 = fn(a2)
        npt.assert_equal(f2, e2)
        
        # in-hi
        a3 = np.array([ 0.0,  0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) - 2.0
        e3 = np.array([-1.0, -0.5,  0.0, 0.5,  1.0, 1.0,  1.0,  1.0, 1.0])
        f3 = fn(a3)
        npt.assert_equal(f3, e3)
        
        # lo-hi
        a4 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0,  0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) - 2.0
        e4 = np.array([-1.0, -1.0,  -1.0, -1.0, -1.0, -0.5,  0.0, 0.5,  1.0, 1.0,  1.0, 1.0,  1.0])
        f4 = fn(a4)
        npt.assert_equal(f4, e4)

class Test_create_convert_linear_abs(unittest.TestCase):
    
    def test_create_convert_linear_abs1(self):
        fn = C.create_convert_linear_abs(0.0, 1.0)
        
        # in
        a1 = np.array([ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
        e1 = np.array([ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
        f1 = fn(a1)
        npt.assert_equal(f1, e1)
        
        # lo-in
        a2 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        e2 = np.array([ 0.0,  0.0,   0.0,  0.0,  0.0, 0.25, 0.5, 0.75, 1.0])
        f2 = fn(a2)
        npt.assert_equal(f2, e2)
        
        # in-hi
        a3 = np.array([ 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        e3 = np.array([ 0.0, 0.25, 0.5, 0.75, 1.0, 1.0,  1.0, 1.0,  1.0])
        f3 = fn(a3)
        npt.assert_equal(f3, e3)
        
        # lo-hi
        a4 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        e4 = np.array([ 0.0,  0.0,   0.0,  0.0,  0.0, 0.25, 0.5, 0.75, 1.0, 1.0,  1.0, 1.0,  1.0])
        f4 = fn(a4)
        npt.assert_equal(f4, e4)
    
    def test_create_convert_linear_abs2(self):
        fn = C.create_convert_linear_abs(1.0, 2.0)
        
        # in
        a1 = np.array([ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]) + 1.0
        e1 = np.array([ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
        f1 = fn(a1)
        npt.assert_equal(f1, e1)
        
        # lo-in
        a2 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]) + 1.0
        e2 = np.array([ 0.0,  0.0,   0.0,  0.0,  0.0, 0.25, 0.5, 0.75, 1.0])
        f2 = fn(a2)
        npt.assert_equal(f2, e2)
        
        # in-hi
        a3 = np.array([ 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) + 1.0
        e3 = np.array([ 0.0, 0.25, 0.5, 0.75, 1.0, 1.0,  1.0, 1.0,  1.0])
        f3 = fn(a3)
        npt.assert_equal(f3, e3)
        
        # lo-hi
        a4 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) + 1.0
        e4 = np.array([ 0.0,  0.0,   0.0,  0.0,  0.0, 0.25, 0.5, 0.75, 1.0, 1.0,  1.0, 1.0,  1.0])
        f4 = fn(a4)
        npt.assert_equal(f4, e4)
    
    def test_create_convert_linear_abs3(self):
        fn = C.create_convert_linear_abs(-1.0, 1.0)
        
        # in
        a1 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        e1 = np.array([ 1.0,  0.75,  0.5,  0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        f1 = fn(a1)
        npt.assert_equal(f1, e1)
        
        # lo-in
        a2 = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
        e2 = np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  0.5, 0.0, 0.5, 1.0])
        f2 = fn(a2)
        npt.assert_equal(f2, e2)
        
        # in-hi
        a3 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        e3 = np.array([ 1.0,  0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0])
        f3 = fn(a3)
        npt.assert_equal(f3, e3)
        
        # lo-hi
        a4 = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        e4 = np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0])
        f4 = fn(a4)
        npt.assert_equal(f4, e4)
    
    def test_create_convert_linear_abs4(self):
        fn = C.create_convert_linear_abs(-2.0, -1.0)
        r = lambda x: list(reversed(x))
        
        # in
        a1 = np.array(  [ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]) - 2.0
        e1 = np.array(r([ 0.0,  0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]))
        f1 = fn(a1)
        npt.assert_equal(f1, e1)
        
        # lo-in
        a2 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]) - 2.0
        e2 = np.array([ 1.0,  1.0,   1.0,  1.0,  1.0, 0.75, 0.5, 0.25, 0.0])
        f2 = fn(a2)
        npt.assert_equal(f2, e2)
        
        # in-hi
        a3 = np.array([ 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) - 2.0
        e3 = np.array([ 1.0, 0.75, 0.5, 0.25, 0.0, 0.0,  0.0, 0.0,  0.0])
        f3 = fn(a3)
        npt.assert_equal(f3, e3)
        
        # lo-hi
        a4 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) - 2.0
        e4 = np.array([ 1.0,  1.0,   1.0,  1.0,  1.0, 0.75, 0.5, 0.25, 0.0, 0.0,  0.0, 0.0,  0.0])
        f4 = fn(a4)
        npt.assert_equal(f4, e4)

class Test_vectorize_custom(unittest.TestCase):
    
    def test_vectorize_custom1(self):
        fn = np.vectorize(lambda x: np.abs(x), otypes=[np.float32])
        a1 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        e1 = np.array([ 1.0,  0.75,  0.5,  0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        f1 = fn(a1)
        npt.assert_equal(f1, e1)
    
    def test_vectorize_custom2(self):
        sn = "abs(x)"
        fn = np.vectorize(eval(f"lambda x: {sn}"), otypes=[np.float32])
        a1 = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        e1 = np.array([ 1.0,  0.75,  0.5,  0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        f1 = fn(a1)
        npt.assert_equal(f1, e1)


if __name__ == '__main__':
    import unittest
    unittest.main()
