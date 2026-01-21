import numpy as np
import colorcet as cc


path='./'
figPath = path+'figures/'
envPath = path+'env/'
dataPath = path+'data/'
runPath = path+'runs/'

str_E = 'white_noise'

strains_subsetE = [1, 2, 6, 9, 11, 15, 18, 20, 21, 24]
strains_subsetK = [25, 26, 29, 34, 37, 38, 39, 41, 43, 45]

strains_subset = strains_subsetE + strains_subsetK

strains = ['C001', 'C002',  'C006',  'C011',  'C012',  'C021',  'C022',  'C031',  'C051',  'C063',  'C094',  'C107',  'C115',  'C131',  'C141',  'C201',  'C227',  'C232',  'C247',  'C261',  'C286',  'C290',  'C302',  'C309',  'C324',  'K037',  'K038',  'K087',  'K094',  'K112',  'K114',  'K125',  'K141',  'K168',  'K177',  'K200',  'K201',  'K209',  'K213',  'K216',  'K224',  'K225',  'K241',  'K248',  'K249',  'K253',  'K257',  'K275',  'K285',  'K300']
strain_names = ["PF_EC01", "PF_EC02", "PF_EC03", "PF_EC04", "PF_EC05", "PF_EC06", "PF_EC07", "PF_EC08", "PF_EC09", "PF_EC10", "PF_EC11", "PF_EC12", "PF_EC13", "PF_EC14", "PF_EC15","PF_EC16", "PF_EC17", "PF_EC18", "PF_EC19", "PF_EC20", "PF_EC21", "PF_EC22", "PF_EC23", "PF_EC24", "PF_EC25", "PF_KPN01", "PF_KPN02", "PF_KPN03", "PF_KPN04", "PF_KQ01", "PF_KPN05", "PF_KPN06", "PF_KPN07", "PF_KPN08", "PF_KPN09", "PF_KQ02", "PF_KQ03", "PF_KPN10", "PF_KPN11", "PF_KQ04", "PF_KV01", "PF_KPN12", "PF_KPN13", "PF_KPN14", "PF_KPN15", "PF_KPN16", "PF_KPN17", "PF_KPN18", "PF_KPN19", "PF_KPN20", "PF_EC01", "PF_EC02", "PF_EC03", "PF_EC04", "PF_EC05", "PF_EC06", "PF_EC07", "PF_EC08", "PF_EC09", "PF_EC10", "PF_EC11", "PF_EC12", "PF_EC13", "PF_EC14", "PF_EC15", "PF_EC16", "PF_EC17", "PF_EC18", "PF_EC19", "PF_EC20", "PF_EC21", "PF_EC22", "PF_EC23", "PF_EC24", "PF_EC25", "PF_KPN01", "PF_KPN02", "PF_KPN03", "PF_KPN04", "PF_KQ01", "PF_KPN05", "PF_KPN06", "PF_KPN07", "PF_KPN08", "PF_KPN09", "PF_KQ02", "PF_KQ03", "PF_KPN10", "PF_KPN11", "PF_KQ04", "PF_KV01", "PF_KPN12", "PF_KPN13", "PF_KPN14", "PF_KPN15", "PF_KPN16", "PF_KPN17", "PF_KPN18", "PF_KPN19", "PF_KPN20"]

plasmids = ['WT','TC']

tot_strains = int(len(strains))
cmap_strains = cc.glasbey_light[:tot_strains]

B0 = 1e6 #Initial bacterial density
T = 24 #Duration of experimental season
S0 = 1.0 #Concentration of imiting resource
extinction_threshold=1.0 #Extinction threshold
alphas=[1e-10, 1e-12] #Antibiotic degradation rate
d=0.1 #Transfer dilution rate

A_max=65536*2 #Maximum antibiotic concentrations=[32768, 256, 1024, 32]

expe_params = {
    'B0': B0, #Initial bacterial density
    'A_max': A_max, #Maximum drug concentration
    'alphas': np.array(alphas), #Antibiotic degradation rate
    'T': T,  # Length of experiment
    'S0': S0,  # Resource concentration
    'd': d,  # Resource concentration
    'extinction_threshold': extinction_threshold,
}
verbose=False


blue = '#129FE4'
red = '#E9666F'
