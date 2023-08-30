'''
 Base class for different gravitational wave sources.
'''
from abc import ABC, abstractmethod
from scipy import signal

class Source(ABC):
    '''
      Base abstract class to operate the data.
      Different data representations and data types have to inherit from this class.
    '''
    def __init__(self, **kwards):
        '''
          config_spec -- path to the configuration file with parameters
          dt -- cadence
        '''
        super().__init__()
        #self._timeseries = None
       
    #@property
    #def timeseries(self):
    #    tsA = self._timeseries[0]
    #    tsE = self._timeseries[1]
    #    tsT = self._timeseries[2]
    #    return tsA, tsE, tsT 
 
    #@timeseries.setter
    #def timeseries(self, value):
    #    self._timeseries = value

    #@abstractmethod
    #def freqwave_AET(self):
    #    '''
    #      A, E, T combinations of the waveform in the frequency domain.
    #    '''
    #    raise NotImplementedError

    #@abstractmethod
    #def timewave_AET(self):
        #'''
        #  A, E, T combinations of the waveform in the time domain.
        #'''
        #raise NotImplementedError

#    def transform(self, tsX):
#        '''
#          config_spec -- yaml file which has the type of the spec and the parameters which correpond to the chosen type
#        '''
#        Axx = []
#    
#        if self.config_spec['transform']['type']  == 'stft':
#           f, t, Axx = signal.stft(
#                              tsX, fs = 1.0/self.dt, 
#                              nperseg  = self.config_spec['transform']['stft']['nperseg'], 
#                              noverlap = self.config_spec['transform']['stft']['noverlap'], 
#                              padded   = self.config_spec['transform']['stft']['padded'], 
#                              window   = self.config_spec['transfrom']['stft']['window'],
#                              detrend  = self.config_spec['transfrom']['stft']['detrend'],
#                              )
#        else:
#            raise ValueError 
#
#
#        plotting = False
#        if plotting == True: 
#            '''
#              Plot amplitude and phase of the transformed sinal.
#            '''
#            #thresh = 8
#            amplitude = np.abs(Sxx)#/Sxx.max()            # volume normalize to max 1
#            phase = np.angle(Sxx)
#            #specgram = np.log10(specgram[4:,:])         # take log
#            #specgram[specgram < -thresh] = -thresh      # set anything less than the threshold as the threshold
#        
#            fig, ax = plt.subplots(1, figsize=(18,10))
#            ax.pcolormesh(t/day, np.log10(f[4:]), amplitude, shading = 'gouraud') # np.log(f[4:])
#            ax.set_ylabel('log-Frequency [Hz]')
#            ax.set_xlabel('Time [days]')
#            plt.savefig("stft_abs.png",  bbox_inches='tight')
# 
#            fig, ax = plt.subplots(1, figsize=(18,10))
#            ax.pcolormesh(t/day, np.log10(f[4:]), phase, shading = 'gouraud') 
#            ax.set_ylabel('log-Frequency [Hz]')
#            ax.set_xlabel('Time [days]') 
#            plt.savefig("stft_phase.png", bbox_inches='tight') 
#     
#        return t, f, Axx





                  
