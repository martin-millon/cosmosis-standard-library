import os
import pickle
from cosmosis.datablock import names, SectionOptions
from hierarc.Likelihood.lens_sample_likelihood import LensSampleLikelihood
from astropy.cosmology import w0waCDM
from astropy.constants import c
from lenstronomy.Cosmo.cosmo_interp import CosmoInterp
import copy


class TDCOSMOlenses:
    '''
    This class reproduces the likelihood from https://arxiv.org/abs/2007.02941 (Fig. 7, purple contours). 
    '''
    def __init__(self, options):
        self.dir_path = os.path.dirname(__file__)

        self.data_sets = options.get_string("data_sets", default='tdcosmo2025')
        self.analysis = options.get_string("analysis", default='tdcosmo2025')
        self._num_distribution_draws = options.get_int("num_distribution_draws", default=200)
        self._distances_computation_module = options.get_string("distances_computation_module", default='astropy')

        # 7 TDCOSMO lenses (TDCOSMO IV)
        file = open(os.path.join(self.dir_path, 'tdcosmo7_likelihood_processed.pkl'), 'rb')
        tdcosmo7_likelihood_processed = pickle.load(file)
        file.close()

        # 33 SLACS lenses with SDSS spectroscopy (TDCOSMO IV) -- OUTDATED, DO NOT USE
        file = open(os.path.join(self.dir_path, 'slacs_sdss_likelihood_processed.pkl'), 'rb')
        slacs_sdss_likelihood_processed = pickle.load(file)
        file.close()

        # 5 SLACS with IFU
        file = open(os.path.join(self.dir_path, 'slacs_ifu_likelihood_processed.pkl'), 'rb')
        slacs_ifu_likelihood_processed = pickle.load(file)
        file.close()

        # TDCOSMO2025 (8 Time-delay lenses)
        file = open(os.path.join(self.dir_path, 'tdcosmo2025_likelihood_processed_const_.pkl'), 'rb')
        tdcosmo2025_likelihood_processed = pickle.load(file)
        tdcosmo2025_likelihood_processed = self.read_kin_correction(tdcosmo2025_likelihood_processed, 'TDCOSMO')
        file.close()

        # here we update each individual lens likelihood configuration with the setting of the Monte-Carlo marginalization over hyper-parameter distributions
        for lens in tdcosmo7_likelihood_processed:
            lens['num_distribution_draws'] = self._num_distribution_draws
        for lens in tdcosmo2025_likelihood_processed:
            lens['num_distribution_draws'] = self._num_distribution_draws
        for lens in slacs_sdss_likelihood_processed:
            lens['num_distribution_draws'] = self._num_distribution_draws
        for lens in slacs_ifu_likelihood_processed:
            lens['num_distribution_draws'] = self._num_distribution_draws

        # ====================
        # TDCOSMO IV or TDCOSMO2025 likelihood
        # ====================

        # hear we build a likelihood instance for the sample of 7 TDCOSMO lenses,
        lens_list = []
        if self.analysis == 'tdcosmo_iv':
            kwargs_global_model = None
            if 'tdcosmo7' in self.data_sets:
                lens_list += tdcosmo7_likelihood_processed
            if 'SLACS_SDSS' in self.data_sets:
                lens_list += slacs_sdss_likelihood_processed
            if 'SLACS_IFU' in self.data_sets:
                lens_list += slacs_ifu_likelihood_processed

            assert len(
                lens_list) > 0, "Data not found ! Add at least one of those 3 data sets 'tdcosmo7', 'SLACS_SDSS' or 'SLACS_IFU'"

        elif self.analysis == 'tdcosmo2025':
            kwargs_global_model =  {'lambda_mst_sampling': True,
                      'lambda_mst_distribution': 'GAUSSIAN',
                      'anisotropy_sampling': True,
                      'sigma_v_systematics': False,
                      'anisotropy_model': 'const',
                      'anisotropy_distribution': 'GAUSSIAN',  # for OM, GOM, use GAUSSIAN_SCALED, for const use GAUSSIAN
                      'alpha_lambda_sampling': True,
                      'anisotropy_parameterization': 'TAN_RAD',
                     }
            if 'tdcosmo2025' in self.data_sets:
                lens_list += tdcosmo2025_likelihood_processed

            assert len(
                lens_list) > 0, "Data not found ! Add the data set 'tdcosmo2025'"
        else:
            raise ValueError("Analysis not recognized. Choose either 'tdcosmo_iv' or 'tdcosmo2025'")

        # choose which likelihood you want here:
        self._likelihood = LensSampleLikelihood(lens_list, kwargs_global_model=kwargs_global_model)

        # choose if you want the full astropy distance calculation or a interpolated version of it (for speed-up)
        self._interpolate_distances_type = 'None'

    def cosmosis_cosmo_2_astropy_cosmo(self, block):
        """

        :param cosmosis_cosmo: cosmosis cosmology object
        :return ~astropy.cosmology equivalent cosmology object
        """
        H0 = block['cosmological_parameters', 'h0'] * 100 #in km/s/Mpc
        om = block['cosmological_parameters', 'omega_m']
        ok = block['cosmological_parameters', 'omega_k']
        ob = block['cosmological_parameters', 'omega_b']
        w0 = block['cosmological_parameters', 'w']
        wa = block['cosmological_parameters', 'wa']
        mnu = block['cosmological_parameters', 'mnu']
        nnu = block['cosmological_parameters', 'nnu']

        ol = 1 - om - ok

        if self._distances_computation_module == 'astropy':
            # we are using standard astropy cosmology for distance computation
            cosmo = w0waCDM(H0=H0, Om0=om, Ode0=ol, Ob0=ob, w0=w0, wa=wa, m_nu=mnu, Neff=nnu)
        elif self._distances_computation_module == 'CosmoInterp':
            # we are using an interpolated version of the standard astropy cosmology (for speed-up)
            cosmo = w0waCDM(H0=H0, Om0=om, Ode0=ol, Ob0=ob, w0=w0, wa=wa, m_nu=mnu, Neff=nnu)
            cosmo = CosmoInterp(cosmo=cosmo, z_stop=5, num_interp=100)
        elif self._distances_computation_module == 'camb':
            # we use the camb distances
            z_bg = block['distances', 'z']
            D_A = block['distances', 'd_A']
            K = ok * c.to('km/s').value * H0  #in Mpc^-2
            cosmo = CosmoInterp(ang_dist_list=D_A, z_list=z_bg, Ok0=ok, K=K)
        else:
            raise ValueError()

        return cosmo

    def likelihood(self, block):
        cosmo = self.cosmosis_cosmo_2_astropy_cosmo(block)

        # here the additional parameters required to evaluate the likelihood in accordance with TDCOSMO IV Table 3
        lambda_mst = block['nuisance_strong_lensing', 'lambda_mst']

        # We will define these parameters in the block in log space because the prior is uniform in log_ space. 
        log_lambda_mst_sigma = block['nuisance_strong_lensing', 'log_lambda_mst_sigma']
        lambda_mst_sigma = 10**log_lambda_mst_sigma
        # a_ani = block['nuisance_strong_lensing', 'a_ani']
        # a_ani_sigma = block['nuisance_strong_lensing', 'a_ani_sigma']
        if self.analysis == 'tdcosmo_iv':
            log_a_ani = block['nuisance_strong_lensing', 'log_a_ani']
            a_ani = 10**log_a_ani
        elif self.analysis == 'tdcosmo2025': # in TDCOSMO2025 we use a linear prior on a_ani, because we now use constant anisotropy models
            a_ani = block['nuisance_strong_lensing', 'a_ani']
            gamma_pl_RXJ1131 = block['nuisance_strong_lensing', 'gamma_pl_RXJ1131']
        
        log_a_ani_sigma = block['nuisance_strong_lensing', 'log_a_ani_sigma']
        a_ani_sigma = 10**log_a_ani_sigma
        
        alpha_lambda = block['nuisance_strong_lensing', 'alpha_lambda']

        kwargs_lens_test = {'lambda_mst': lambda_mst,  # mean in the internal MST distribution
                            'lambda_mst_sigma': lambda_mst_sigma,  # Gaussian sigma of the distribution of lambda_mst
                            'alpha_lambda': alpha_lambda,  # slope of lambda_mst with r_eff/theta_E
                            }
        kwargs_kin_test = {'a_ani': a_ani,  # mean a_ani anisotropy parameter in the OM model or constant model
                           'a_ani_sigma': a_ani_sigma,  # sigma(a_ani)⟨a_ani⟩ is the 1-sigma Gaussian scatter in a_ani
                           }
        if self.analysis == 'tdcosmo2025':
            kwargs_lens_test['gamma_pl_list'] = [gamma_pl_RXJ1131] #adding the gamma_pl for RXJ1131 only, as in TDCOSMO2025

        logl = self._likelihood.log_likelihood(cosmo=cosmo, kwargs_lens=kwargs_lens_test, kwargs_kin=kwargs_kin_test)

        return float(logl)

    def read_kin_correction(self, likelihood_list_selected, sample_name):
        if sample_name == "SLACS_IFU":
            file_name = os.path.join(self.dir_path, "kin_axi_jam_scaling/kcwi_correction.pickle")
        elif sample_name == "SL2S":
            file_name = os.path.join(self.dir_path,"kin_axi_jam_scaling/sl2s_correction.pickle")
        elif sample_name == "TDCOSMO":
            file_name = os.path.join(self.dir_path,"kin_axi_jam_scaling/tdcosmo_correction.pickle")
        else:
            raise ValueError("Sample name for kinematic correction not recognized. Choose either 'SLACS_IFU', 'SL2S' or 'TDCOSMO'")

        with open(file_name, "rb") as f:  # read in pre-saved correction file
            jam_scaling = pickle.load(f)

        likelihood_list_new = copy.deepcopy(likelihood_list_selected)

        name_list = [x["name"] for x in likelihood_list_new]
        for name in name_list:
            if name not in [x["name"] for x in jam_scaling]:
                print("no axisymmetric jam scaling for lens %s" % name)
            else:
                pos = name_list.index(name)
                correction = [
                    x["correction_combined"] for x in jam_scaling if x["name"] == name
                ]
                likelihood_list_new[pos]["vel_disp_scaling_distributions"] = correction[0]

        return likelihood_list_new


def setup(options):
    options = SectionOptions(options)
    return TDCOSMOlenses(options)


def execute(block, config):
    like = config.likelihood(block)
    block[names.likelihoods, "TDCOSMO_like"] = like
    return 0

