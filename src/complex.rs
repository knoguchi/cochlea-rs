//! Complex number operations for cochlear model filters.
//!
//! Ported from complex.hpp/complex.c in the original Zilany 2014 implementation.

use std::ops::{Add, Mul, Sub};

/// Complex number with f64 components.
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    /// Create a new complex number.
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Create a complex number from polar form: r * e^(i*theta).
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Create e^(i*theta).
    #[inline]
    pub fn exp_i(theta: f64) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    /// Conjugate of the complex number.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared magnitude (norm).
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude (absolute value).
    #[inline]
    pub fn abs(self) -> f64 {
        self.norm_sqr().sqrt()
    }

    /// Real part.
    #[inline]
    pub fn real(self) -> f64 {
        self.re
    }

    /// Imaginary part.
    #[inline]
    pub fn imag(self) -> f64 {
        self.im
    }

    /// Multiply by a scalar.
    #[inline]
    pub fn scale(self, scalar: f64) -> Self {
        Self {
            re: self.re * scalar,
            im: self.im * scalar,
        }
    }

    /// Complex division.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        let d = other.norm_sqr();
        Self {
            re: (self.re * other.re + self.im * other.im) / d,
            im: (self.im * other.re - self.re * other.im) / d,
        }
    }
}

impl Add for Complex {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f64) -> Self {
        self.scale(scalar)
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;

    #[inline]
    fn mul(self, c: Complex) -> Complex {
        c.scale(self)
    }
}

/// Sum of three complex numbers.
#[inline]
pub fn comp3sum(a: Complex, b: Complex, c: Complex) -> Complex {
    Complex {
        re: a.re + b.re + c.re,
        im: a.im + b.im + c.im,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, 2.0);

        // Magnitude
        assert!((a.abs() - 5.0).abs() < 1e-10);

        // Addition
        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        // Multiplication
        let prod = a * b;
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);

        // Conjugate
        let conj = a.conj();
        assert!((conj.re - 3.0).abs() < 1e-10);
        assert!((conj.im - (-4.0)).abs() < 1e-10);

        // exp_i
        let e = Complex::exp_i(std::f64::consts::PI / 2.0);
        assert!(e.re.abs() < 1e-10);
        assert!((e.im - 1.0).abs() < 1e-10);
    }
}
