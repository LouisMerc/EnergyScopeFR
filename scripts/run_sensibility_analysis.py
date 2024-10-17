import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root directory to sys.path
sys.path.append(project_root)
import my_energyscope as es

analysis_sensibility_only = False

if analysis_sensibility_only == False:
    
    # define project path
    project_path = Path(__file__).parents[1]
    
    # define the parameter to do a variation on
    # either using an excel file or 
    # numbers define in a list

    import_fuel_price = [i for i in range (50, 505, 5)]*7
    share_emission_reduction = []
    [share_emission_reduction.extend([j-10] * 91) for j in range(0, 350, 50)]

    for case_number in range(0, 638):
        print('--------------------------', case_number, '---------------------------------')
        analysis_only= False
        compute_TDs = False
        # loading the config file into a python dictionnary
        config = es.load_config(config_fn='config_ref.yaml', project_path=project_path) 
        config['case_studies'] = os.path.join(project_path,'case_studies', 'paper1', 'import_analysis', 'jf_import_price_emi_reduc')#, )##
        config['case_study'] = 'scenario_' + str(case_number+1)
        config['Working_directory'] = os.getcwd() # keeping current working directory into config
        config['print_hourly_data'] = False
        
        # Reading the data of the csv
        es.import_data(config)

        ## Change the value in the data

        # For analysis on the quantity imported in relation with import price and share of emission reduction  
        config['all_data']['Resources'].loc['JETFUEL_RE', 'avail'] = 910000
        config['all_data']['Resources'].loc['JETFUEL_RE', 'c_op'] = import_fuel_price[case_number]/1000
        config['all_data']['Layers_in_out'].loc['JETFUEL_RE', 'CO2_ATMOSPHERE'] = -share_emission_reduction[case_number]*0.26/100
        


        if not analysis_only:
            if compute_TDs:
                es.build_td_of_days(config)

            # Printing the .dat files for the optimisation problem       
            es.print_data(config)
            filename = os.path.join(config['case_studies'], config['case_study'], "not_working.txt")
            if os.path.exists(filename):
                # If the file exists, delete it
                os.remove(filename)
            # Running EnergyScope
            run_ended = es.run_es_sensi(config)
            config['print_sankey'] = False
            if run_ended == False:           
                file = open(filename, "w")
                file.close()