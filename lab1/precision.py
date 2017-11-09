import numpy as np


floating_point_types = {
    # sign bit, 5 bits exponent, 10 bits mantissa
    'half': np.float16,

    # sign bit, 8 bits exponent, 23 bits mantissa
    'single': np.float32,

    # sign bit, 11 bits exponent, 52 bits mantissa
    'double': np.float64,

    # NumPy does not provide a dtype with more precision than C long double``s
    # For efficient memory alignment, np.longdouble is usually stored
    # padded with zero bits, typically:
    # - on 32-bit systems they are padded to 96 bits,
    # - on 64-bit systems they are padded to 128 bits.

    # In spite of the names, np.float96 and np.float128
    # provide only as much precision as np.longfloat,
    # that is, 80 bits on most x86 machines
    # and 64 bits in standard Windows builds.

    # sign bit, 15 bits exponent, 64 bits mantissa
    'extended': np.longfloat,
}

number = floating_point_types['extended']


def precision_test():
    for num in floating_point_types.values():
        print(np.dtype(num))
        print(np.finfo(num))

        z = x = num(3.14159265358979)
        for _ in range(50):
            z = num(z * x)

        print('{:10.10f}'.format(z))
        print()


if __name__ == "__main__":
    precision_test()
