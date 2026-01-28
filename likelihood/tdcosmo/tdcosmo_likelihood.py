import os
import pickle
import hierarc
from cosmosis.datablock import names, SectionOptions
from hierarc.Likelihood.lens_sample_likelihood import LensSampleLikelihood
from astropy.cosmology import w0waCDM
from astropy.constants import c
from lenstronomy.Cosmo.cosmo_interp import CosmoInterp
import numpy as np
import pandas as pd
import copy
import warnings
from packaging import version
hierarc_version = hierarc.__version__

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

        if self.analysis == 'tdcosmo_iv':
            # TDCOSMO IV data sets
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

            # here we update each individual lens likelihood configuration with the setting of the Monte-Carlo marginalization over hyper-parameter distributions
            for lens in tdcosmo7_likelihood_processed:
                lens['num_distribution_draws'] = self._num_distribution_draws
            for lens in slacs_sdss_likelihood_processed:
                lens['num_distribution_draws'] = self._num_distribution_draws
            for lens in slacs_ifu_likelihood_processed:
                lens['num_distribution_draws'] = self._num_distribution_draws

        elif self.analysis == 'tdcosmo2025':
            # TDCOSMO 2025 data sets
            # TDCOSMO2025 (8 Time-delay lenses)
            file = open(os.path.join(self.dir_path, 'tdcosmo2025_likelihood_processed_const_.pkl'), 'rb')
            tdcosmo2025_likelihood_processed = pickle.load(file)
            file.close()

            # SLACS with KCWI from the TDCOSMO 2025 analysis (11 SLACS )
            slacs_kcwi_likelihood = 'slacs_kcwi_const_processed.pkl'
            file = open(os.path.join(self.dir_path, slacs_kcwi_likelihood), 'rb')
            slacs_kcwi_likelihood_processed = pickle.load(file)
            file.close()

            # SL2S from the TDCOSMO 2025 analysis (4 lenses)
            sl2s_likelihood = 'sl2s_const_processed_all.pkl'
            file = open(os.path.join(self.dir_path, sl2s_likelihood), 'rb')
            sl2s_likelihood_processed = pickle.load(file)
            file.close()

            #apply quality cut
            slacs_kcwi_likelihood_processed = self.quality_cut(slacs_kcwi_likelihood_processed)
            sl2s_likelihood_processed = self.quality_cut(sl2s_likelihood_processed)

            #add kaapa_ext correction
            slacs_kcwi_likelihood_processed = self.add_kappa_dist(slacs_kcwi_likelihood_processed, 'SLACS')
            sl2s_likelihood_processed = self.add_kappa_dist(sl2s_likelihood_processed, 'SL2S')

            lens_selected_slacs_kcwi = ['SDSSJ0029-0055', 'SDSSJ0037-0942', 'SDSSJ1112+0826', 'SDSSJ1204+0358', 'SDSSJ1250+0523',
                                   'SDSSJ1306+0600', 'SDSSJ1402+6321', 'SDSSJ1531-0105', 'SDSSJ1621+3931', 'SDSSJ1627-0053',
                                   'SDSSJ1630+4520']

            lens_selected_sl2s = ['SL2SJ0226-0420', 'SL2SJ0855-0147', 'SL2SJ0904-0059', 'SL2SJ2221+0115']

            lens_selected_tdcosmo2025 = ['B1608+656', 'RXJ1131-1231', 'HE0435-1223', 'SDSS1206+4332', 'WFI2033-4723',
                                     'PG1115+080', 'DES0408-5354', 'WGD2038-4008']

            #apply the selection
            slacs_kcwi_likelihood_processed = self.selected_likelihood(lens_selected_slacs_kcwi, slacs_kcwi_likelihood_processed)
            sl2s_likelihood_processed = self.selected_likelihood(lens_selected_sl2s, sl2s_likelihood_processed)
            tdcosmo2025_likelihood_processed = self.selected_likelihood(lens_selected_tdcosmo2025, tdcosmo2025_likelihood_processed)

            #apply axi symetric JAM kinematic correction
            tdcosmo2025_likelihood_processed = self.read_kin_correction(tdcosmo2025_likelihood_processed, 'TDCOSMO')
            sl2s_likelihood_processed = self.read_kin_correction(sl2s_likelihood_processed, 'SL2S')
            slacs_kcwi_likelihood_processed = self.read_kin_correction(slacs_kcwi_likelihood_processed, 'SLACS_KCWI')

            for lens in tdcosmo2025_likelihood_processed:
                lens['num_distribution_draws'] = self._num_distribution_draws
            for lens in slacs_kcwi_likelihood_processed:
                lens['num_distribution_draws'] = self._num_distribution_draws
            for lens in sl2s_likelihood_processed:
                lens['num_distribution_draws'] = self._num_distribution_draws


        # ====================
        # TDCOSMO IV or TDCOSMO2025 likelihood
        # ====================

        # hear we build a likelihood instance for the sample of 7 TDCOSMO lenses,
        lens_list = []
        if self.analysis == 'tdcosmo_iv':
            warnings.warn(
                "TDCOSMO IV is outdated. Use TDCOSMO2025 instead.",
                category=DeprecationWarning,
                stacklevel=2
            )
            if version.parse(hierarc_version) >= version.parse("1.2.0"):
                raise ValueError("TDCOSMO IV analysis is not compatible with hierarc versions >= 1.2.0. Please use TDCOSMO2025 analysis instead or revert hierarc to version 1.1.2.")

            if 'tdcosmo7' in self.data_sets:
                lens_list += tdcosmo7_likelihood_processed
            if 'SLACS_SDSS' in self.data_sets:
                warnings.warn("SLACS_SDSS data set is outdated, do not use it. Use SLACS_IFU or the TDCOSMO2025 data set instead.",
                              category=DeprecationWarning, stacklevel=2)
                lens_list += slacs_sdss_likelihood_processed
            if 'SLACS_IFU' in self.data_sets:
                lens_list += slacs_ifu_likelihood_processed

            assert len(
                lens_list) > 0, "Data not found ! Add at least one of those 3 data sets 'tdcosmo7', 'SLACS_SDSS' or 'SLACS_IFU'"
            self._likelihood = LensSampleLikelihood(lens_list)

        elif self.analysis == 'tdcosmo2025':
            if version.parse(hierarc_version) < version.parse("1.2.0"):
                raise ValueError("TDCOSMO2025 analysis is only compatible with hierarc versions >= 1.2.0")
            kwargs_global_model =  {'lambda_mst_sampling': True,
                      'lambda_mst_distribution': 'GAUSSIAN',
                      'anisotropy_sampling': True,
                      'sigma_v_systematics': False,
                      'anisotropy_model': 'const',
                      'anisotropy_distribution': 'GAUSSIAN',  # for OM, GOM, use GAUSSIAN_SCALED, for const use GAUSSIAN
                      'alpha_lambda_sampling': True,
                      'anisotropy_parameterization': 'TAN_RAD',
                     }
            self.n_slacs_kcwi = 0
            if 'tdcosmo2025' in self.data_sets:
                lens_list += tdcosmo2025_likelihood_processed
            if 'SLACS_KCWI' in self.data_sets:
                self.n_slacs_kcwi = len(slacs_kcwi_likelihood_processed)
                lens_list += slacs_kcwi_likelihood_processed
            if 'SL2S' in self.data_sets:
                lens_list += sl2s_likelihood_processed

            assert len(
                lens_list) > 0, "Data not found ! Add the data set 'tdcosmo2025'"
            self._likelihood = LensSampleLikelihood(lens_list, kwargs_global_model=kwargs_global_model)
        else:
            raise ValueError("Analysis not recognized. Choose either 'tdcosmo_iv' or 'tdcosmo2025'")


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
            if 'SLACS_KCWI' in self.data_sets:
                gamma_pl_list_SLACS_KCWI = []
                for i in range(self.n_slacs_kcwi):
                    gamma_pl_list_SLACS_KCWI.append(block['nuisance_strong_lensing', f'gamma_pl_SLACS_KCWI{i}'])
        
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
            if 'SLACS_KCWI' in self.data_sets:
                kwargs_lens_test['gamma_pl_list'] += gamma_pl_list_SLACS_KCWI

        logl = self._likelihood.log_likelihood(cosmo=cosmo, kwargs_lens=kwargs_lens_test, kwargs_kin=kwargs_kin_test)

        return float(logl)

    def read_kin_correction(self, likelihood_list_selected, sample_name):
        if sample_name == "SLACS_KCWI":
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

    def quality_cut(self, likelihood_list, use_quality_data_only = True):
        """whether or not to use only the lenses with lens models of imaging data or with ifu spectra

        Args:
            likelihood_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        likelihood_list_cut = []

        for likelihood in likelihood_list:
            if 'flag_imaging' in likelihood:
                flag_imaging = copy.deepcopy(likelihood['flag_imaging'])
                del likelihood['flag_imaging']
            else:
                flag_imaging = 1
            if 'flag_ifu' in likelihood:
                flag_ifu = copy.deepcopy(likelihood['flag_ifu'])
                del likelihood['flag_ifu']
            else:
                flag_ifu = 1

            if use_quality_data_only is True:
                if flag_imaging < 1 or flag_ifu < 1:
                    pass
                else:
                    likelihood_list_cut.append(likelihood)
            else:
                likelihood_list_cut.append(likelihood)
        return likelihood_list_cut

    def add_kappa_dist(self, likelihood_list, sample_name):
        """add external kappa distribution for the SL2S and the SLACS sample

        Args:
            likelihood_list (_type_): likelihood list
            sample_name (_type_): 'SLACS' or 'SL2S'

        Returns:
            _type_: updated likelihood list
        """
        lens_name_list = [x['name'] for x in likelihood_list]
        list_kappa_ext = []
        lens_list = []

        if sample_name == 'SLACS':
            kappa_choice_ending = '_computed_1innermask_nobeta_zgap-1.0_-1.0_fiducial_120_gal_120_oneoverr_23.0_med_increments2_2_emptymsk.cat'
            kappa_bins = np.linspace(-0.05, 0.2, 50)

            for name in lens_name_list:
                try:
                    filepath = os.path.join(self.dir_path, 'ExternalLenses/SLACS/kappa_ext/', 'kappahist_'+name+kappa_choice_ending)
                    output = np.loadtxt(filepath, delimiter=' ', skiprows=1)
                    kappa_sample = output[:, 0]
                    kappa_weights = output[:, 1]
                    kappa_pdf, kappa_bin_edges = np.histogram(kappa_sample, weights=kappa_weights, bins=kappa_bins, density=True)
                    list_kappa_ext.append({'los_distribution_individual': 'PDF', 'kwargs_los_individual': {'bin_edges': kappa_bin_edges, 'pdf_array': kappa_pdf}})
                    lens_list.append(name)
                except:
                    print('lens %s does not have a kappa_ext file %s' % (name, filepath))
        elif sample_name == 'SL2S':
            df_sl2s_los_gev = os.path.join(self.dir_path, 'ExternalLenses/SL2S/kappa_ext/', 'sl2s_los_gev.csv')
            df_sl2s_los_gev = pd.read_csv(df_sl2s_los_gev) # read in sl2s los data with GEV fit
            log_sigma_kappa_ext = df_sl2s_los_gev['log_sigma_kext'].values
            xi_kappa_ext = df_sl2s_los_gev['xi_kext'].values
            mu_kappa_ext = df_sl2s_los_gev['mu_kext'].values
            name_gev_los = list(df_sl2s_los_gev['name'].values)

            for name in lens_name_list:
                if name in name_gev_los:
                    pos = name_gev_los.index(name)
                    lens_list.append(name)
                    list_kappa_ext.append({'kwargs_los_individual': {'mean': mu_kappa_ext[pos], 'xi': xi_kappa_ext[pos], 'sigma': np.exp(log_sigma_kappa_ext[pos])}, 'los_distribution_individual': 'GEV'})
                else:
                    print('lens %s does not have a kappa_ext distribution' % name)

        likelihood_list_new = []
        for i, name in enumerate(lens_list):
            pos = lens_name_list.index(name)
            kwargs_lens = likelihood_list[pos]
            kwargs_lens['kwargs_los_individual'] = list_kappa_ext[i]['kwargs_los_individual']
            kwargs_lens['los_distribution_individual'] = list_kappa_ext[i]['los_distribution_individual']
            likelihood_list_new.append(kwargs_lens)
        return likelihood_list_new

    def selected_likelihood(self, lens_list_selected, likelihood_list):
        """select the likelihoods of selected lenses

        Args:
            lens_list_selected (_type_): list of lens names
            likelihood_list (_type_): list of lens likelihoods containing selected lenses

        Returns:
            _type_: list of lens likelihoods of selected lenses
        """
        likelihood_new = []
        name_all = [x['name'] for x in likelihood_list]
        for name in lens_list_selected:
            try:
                pos = name_all.index(name)
                likelihood_new.append(likelihood_list[pos])
            except ValueError:
                pass
        print('selected lens sample has {} lenses'.format(len(likelihood_new)))
        return likelihood_new


def setup(options):
    options = SectionOptions(options)
    return TDCOSMOlenses(options)


def execute(block, config):
    like = config.likelihood(block)
    block[names.likelihoods, "TDCOSMO_like"] = like
    return 0

