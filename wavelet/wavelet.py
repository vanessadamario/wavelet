import numpy as np
import meyer

class wavelet(object):

    # actually the only available choice in wavelet_type='meyer'

    def __init__(self, wavelet_type='meyer', dyadic_exp=2.):

        self.wavelet_type = wavelet_type
        self.dyadic_exp = dyadic_exp

        if wavelet_type is not 'meyer':
            print ('sorry, work in progress')

    def dyadic_scales(self, x, j):
        return pow(2,-x/j)

    def psi(self, npoints, nscale) :

        """

        Parameters
        ----------
        npoints: int
            Number of sampling points (length of the signal)

        nscale: int
            Number of scale

        Returns
        -------
        mother_funct_different_scales: array_like, shape = (npoints, nscale)
            Matrix of mother wavelet fourier transform for fixed values of the scale

        scale: array_like, shape = (nscale, )
            Values of the scale

        """

        if ( not(np.isscalar(npoints)) | (not(np.isscalar(nscale))) ):
            exit(-1)

        if nscale >= 2 :
            nscale = int(np.floor(np.log2(npoints)))
        scale = np.zeros((nscale+1,))
        scale[0] = 1

        l_orig = npoints # creates asymmetric interval
        if np.mod(npoints,2) == 0:
            npoints+=1

        X = self.dyadic_scales(-nscale+1, self.dyadic_exp) # maximum scale, min frequency

        xi_x_init = np.linspace(0,X,(npoints+1)/2)
        xi_x = np.concatenate((-xi_x_init[-1:-len(xi_x_init):-1], xi_x_init))
        psi = np.zeros((npoints, nscale + 1), dtype=np.complex_)
        mother_funct_different_scales = np.zeros((npoints, nscale+1))

        if(self.wavelet_type == 'meyer'):
            mother_funct_different_scales[:,0] = meyer.scaling(xi_x)
            for j in range(0, nscale):
                a = self.dyadic_scales(j, self.dyadic_exp);
                scale[j+1] = a
                x = a * xi_x
                mother_funct_different_scales[:,j+1] = meyer.mother(x) * 1/np.sqrt(a)

        mother_funct_different_scales = mother_funct_different_scales[0:l_orig, :]

        return mother_funct_different_scales , scale

    def cwt(self, x, n_scales):

        """

        Parameters
        ----------
        x: array_like, shape = (npoints,)
            signal

        n_scales: int, optional
            Number of scale

        Returns
        -------
        coefficients: array_like, shape = (npoints, nscale)
            Matrix of wavelet coefficients evaluated at each point

        scale: array_like, shape = (nscale, )
            Values of the scale

        """
        signalFFT = 1./len(x) * np.fft.fftshift(np.fft.fft(x))
        matrixFFT = np.zeros((len(x), n_scales), dtype = np.complex_)
        for i in range(0,n_scales-1):
            matrixFFT[:,i] = signalFFT

        [motherf, scales] = self.psi(len(x), n_scales)
        product = np.multiply(matrixFFT, motherf)
        coefficient = len(x) * np.fft.ifft(np.fft.ifftshift(product, axes = 0), axis = 0)

        for j in range (0, n_scales-1):
            coefficient[:,j] = coefficient[:,j] * scales[j]
        return coefficient, scales

    def icwt(self, coef):

        """

        Parameters
        ----------
        coef: array_like, shape = (npoints, n_scales)
            wavelet coefficients calculated using self.cwt

        Returns
        -------
        recons_signal: array_like, shape = (npoints, )
            reconstruction of the signal using the inverse wavelet transform

        """
        [motherf, scales] = self.psi(coef.shape[0], coef.shape[1])

        for j in range (0, coef.shape[1]-1):
            coef[:,j] = coef[:,j]/scales[j]

        coefFFT = 1./coef.shape[0] * np.fft.fftshift(np.fft.fft(coef, axis = 0), axes = 0)
        product_i = np.multiply(coefFFT, motherf)
        sum_on_scales = np.sum(product_i, axis=1)
        recons_signal = coef.shape[0] * np.fft.ifft(np.fft.ifftshift(sum_on_scales))

        return recons_signal
