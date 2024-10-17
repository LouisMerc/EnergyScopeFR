import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

from my_energyscope import elec_order_graphs, plotting_names, rename_storage_power, colors_elec
import my_energyscope as es

#This file gather the important functions for the analyze of the results at the output of the model 

def plot_layer_balance_td (layer_balance: pd.DataFrame, title='Layer electricity', number_tds = 12) :
    """
    Plot the layer balance for the different typical days
    The type of layer is contained in layer balance
    -Input : -layer_balance : Hourly dataframe with producing (>0) and consumming (<0) technologies (columns) at each hour (rows), is a direct output of the read_outputs function
             -Title : str, title of the graph
             -number_tds : number of typical days    
    -Output : -A figure of the hourly balance for the considered layer and the different typical days 
    """
    plotdata = layer_balance.copy()
    plotdata = plotdata.fillna(0)
    #Modify data to supress values lower than 0
    colonnes_zeros = plotdata.columns[plotdata.abs().eq(0).all()]
    
    plotdata = plotdata.drop(colonnes_zeros, axis=1)
    colonnes_zeros = plotdata.columns[plotdata.abs().sum() < 100]
    plotdata = plotdata.drop(colonnes_zeros, axis=1)

    names_column = plotdata.columns
    plotdata = plotdata.reset_index()
    plotdata[' Time'] = plotdata[' Time'] + 24 * (plotdata['Td ']-1)

    #Adjust the values for the storage technologies
    column_storage_out = plotdata.columns[plotdata.columns.str.endswith('_Pout')]
    column_storage_in = plotdata.columns[plotdata.columns.str.endswith('_Pin')]
    prefix = column_storage_out.str.split('_Pout').str[0].tolist()

    dfin = pd.DataFrame()
    dfin[prefix] = plotdata[column_storage_in]
    dfout = pd.DataFrame()
    dfout[prefix] = plotdata[column_storage_out]
    plotdata[prefix] = dfin + dfout

    names_column = names_column.difference(column_storage_in).difference(column_storage_out).insert(0, prefix)
    # Add a specific color for every technology, the colors are taken from the techno_color.csv file 
    techno_color = pd.read_csv(os.path.join(os.getcwd(), 'my_energyscope', 'postprocessing', 'draw_sankey',"techno_color.csv"), index_col=0, sep=';').reset_index()
    COLOR_node = [techno_color[techno_color['Name']==i.replace(' ', '')]['Color_bar_plot_1'].item() for i in names_column]
    fig = px.bar(plotdata, x=' Time' ,y=names_column, color_discrete_sequence=COLOR_node)
    colors = ["rgba(230, 25, 75, 0.2)", "rgba(60, 180, 75, 0.2)", "rgba(255, 225, 25, 0.2)",
          "rgba(0, 130, 200, 0.2)", "rgba(245, 130, 48, 0.2)", "rgba(145, 30, 180, 0.2)"]
    
    # Add color values behind for each days, add lines between each days, add a line on y=0 
    x_ranges = [(i, i+24) for i in range(0, (number_tds-1)*24+1, 24)]
    for i, (x_start, x_end) in enumerate(x_ranges):
        color = colors[i % len(colors)]  # Select the corresponding color from the list in a loop.
        fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=x_start,
        y0=0,
        x1=x_end,
        y1=1,
        fillcolor=color,
        layer="below",
        line=dict(color="rgba(0, 0, 0, 0)"),
        )
    time_intervals = [i for i in range(24, (number_tds)*24+1, 24)]
    fig.add_shape(
                type="line",
                x0=0,x1=1,
                y0=0, y1=0,
                xref='paper', yref='y',
                line=dict(color="black", width=1)
            )
    for interval in time_intervals:
        fig.add_shape(
                type="line",
                x0=interval,x1=interval,
                y0=0, y1=1,
                xref='x', yref='paper',
                line=dict(color="black", width=1)
            )
        
    # Replace the x_axis with the different typical days
    custom_ticks = [12 + 24*i for i in range(0, number_tds)]
    custom_tick_labels = ['TD {}'.format(i) for i in range(1, number_tds+1)]
    fig.update_layout(xaxis_title= "Time (hours)", yaxis_title="Energy production/consumption (GW)"
                      , xaxis=dict(tickmode='array',tickvals=custom_ticks,ticktext=custom_tick_labels), font=dict(size=20), title=title)
    fig.update_layout(font=dict(size=16))
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))
    fig.update_layout(width=2000, height=1200)
    return(fig)

def plot_total_cost_system (outputs):
    """
    Plot the share of the different technologies in the total cost of the system under the form of a pie chart
    -Input :  -outputs : the file that contains all data concerning a specific run of energyscope, is the direct results of the read_outputs function
    -Output : -A figure of a pie chart of the share of different technology in the total cost of the system 
    """
    cost_out = outputs['cost_breakdown']
    dict_cost, a = {}, 0
    for index, row in cost_out.iterrows():
        cost = int(cost_out.loc[index, 'C_inv'] + cost_out.loc[index, 'C_maint'] + cost_out.loc[index, 'C_op'])
        dict_cost[a] = (index, cost)
        a = a+1
    df = pd.DataFrame.from_dict(dict_cost, orient='index', columns=[ 'Tech', 'Value'])
    df = df[df['Value'].ge(100)]
    fig = px.pie(df, values='Value', names='Tech')
    fig.update_traces(textposition='inside', textinfo='percent+label+value')
    fig.update_layout(title_text='Total yearly cost of French energy system; Total = '+str(df['Value'].sum())+' MEUR/year')
    return(fig)

def plot_share_ghg_construction (outputs):
    """
    Plot the share of the different technologies in the total GHGs due to the construction under the form of a pie chart
    -Input :  -outputs : the file that contains all data concerning a specific run of energyscope, is the direct results of the read_outputs function 
    -Output : -A figure of a pie chart of the share of different technology in the total GHG emitted due to construction
    """
    df = outputs['gwp_breakdown']
    df = df[df['GWP_constr'].ge(100)]
    df = df.reset_index()
    fig = px.pie(df, values='GWP_constr', names='Name')
    fig.update_traces(textposition='inside', textinfo='percent+label+value')
    fig.update_layout(title_text='Total yearly emission of GHG from construction; Total = '+str(int(df['GWP_constr'].sum()))+' ktCO2eq/year')
    return(fig)

def plot_energy_stored(energy_stored: pd.DataFrame):
    """
    Plot the evolution of the energy stored with different energy vectors over the entire year
    -Input :  -energy_stored : A dataframe with the value of the energy stored for each hour of the year (rows) 
    for all the different storage technologies (columns), is an outputs of the read_outputs function  
    -Output : -A figure of the evolution of the quantity of energy stored over the entire year 
    """
    print(energy_stored)
    plotdata = energy_stored.copy()

    plotdata = plotdata.fillna(0)
    colonnes_zeros = plotdata.columns[plotdata.eq(0).all()]
    plotdata = plotdata.drop(colonnes_zeros, axis=1)
    col_sum = plotdata.abs().sum()

    # Supress vectors with not enough energy stored
    threshold = 8760
    cols_to_drop = col_sum[col_sum < threshold].index
    plotdata = plotdata.drop(cols_to_drop, axis=1)

    names_column = plotdata.columns
    plotdata = plotdata.reset_index()
    # Select a value every 24 hours to reduce the total size of the graph
    plotdata = plotdata[::24]
    column_storage_out = plotdata.columns[plotdata.columns.str.endswith('_out')]
    column_storage_in = plotdata.columns[plotdata.columns.str.endswith('_in')]
    names_column = names_column.difference(column_storage_in).difference(column_storage_out)
    plotdata[names_column] = plotdata[names_column] - plotdata[names_column].min(axis=0)
    # Add a specific color for every technology, the colors are taken from the techno_color.csv file 
    techno_color = pd.read_csv(os.path.join(os.getcwd(), 'my_energyscope', 'postprocessing', 'draw_sankey',"techno_color.csv"), index_col=0, sep=';').reset_index()
    COLOR_node = [techno_color[techno_color['Name']==i.replace(' ', '')]['Color_vector'].item() for i in names_column]
    fig = px.bar(plotdata, x='Time' ,y=names_column, color_discrete_sequence= COLOR_node)
    fig.update_layout(xaxis_title= "Time (hours)", yaxis_title="Energy stored (GWh)")
    fig.update_layout(font=dict(size=18))
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))
    fig.update_layout(width=2000, height=1200)
    return(fig)

def compute_load_factors(outputs):
    """
    Print and return the load factor of different technologies
    -Input :  -outputs : the file that contains all data concerning a specific run of energyscope, is the direct results of the read_outputs function 
    -Output : -value of the load factors for different technologies    
    """
    alk_ely_load_factor = -outputs['year_balance'].loc['ALKALINE_ELECTROLYSIS', 'ELECTRICITY']/(outputs['assets'].loc['ALKALINE_ELECTROLYSIS', 'f']*8760)
    PV_load_factor = outputs['year_balance'].loc['PV', 'ELECTRICITY']/(outputs['assets'].loc['PV', 'f']*8760)
    WIND_ONSHORE_load_factor = outputs['year_balance'].loc['WIND_ONSHORE', 'ELECTRICITY']/(outputs['assets'].loc['WIND_ONSHORE', 'f']*8760)
    WIND_OFFSHORE_load_factor = outputs['year_balance'].loc['WIND_OFFSHORE', 'ELECTRICITY']/(outputs['assets'].loc['WIND_OFFSHORE', 'f']*8760)
    print('Alkaline Electrolysis load factor (%) :', alk_ely_load_factor)
    print('PV panels load factor (%) :', PV_load_factor)
    print('Onshore wind load factor (%) :', WIND_ONSHORE_load_factor)
    print('Offshore wind load factor (%) :', WIND_OFFSHORE_load_factor)
    return(alk_ely_load_factor, PV_load_factor, WIND_ONSHORE_load_factor, WIND_OFFSHORE_load_factor)



##################### For sensibility analysis ###########################

def read_data_post_process(directory_path, hourly_read, name_folder_scenario):
    """
    Store data for different runs in a single list
    Input : -directory_path : the path of the folder where the different runs considered are stored
            -hourly_read : boolean True if the runs include hourly data; False if not
            -name_folder_scenario : boolean True if the name of the different runs is scenario_k with k a natural number; False else
    Return : A list with each element of the list being the outputs of a run of energyscope
    """
    L = []   
    
    for j, folder_name in enumerate(os.listdir(directory_path)):
        if name_folder_scenario == True:
            folder_name = 'scenario_'+str(j+1)
        # define project path
        project_path = Path(__file__).parents[2]

        # loading the config file into a python dictionnary
        config = es.load_config(config_fn='config_ref.yaml', project_path=project_path)
        config['Working_directory'] = os.getcwd() # keeping current working directory into config
        config['case_studies'] = directory_path
        config['case_study'] = folder_name

        print(folder_name)
        # Reading outputs
        if os.path.exists(os.path.join(config['case_studies'], config['case_study'], 'not_working.txt')) == True: 
            outputs['folder_name'] = folder_name
            L.append(False)
        else:
            outputs = es.read_outputs(config, hourly_data=hourly_read)
            outputs['folder_name'] = folder_name
            L.append(outputs)
    return(L)


def file_compute_parameters(directory_path, hourly_read, name_folder_scenario):
    """Store data for different runs in a single dataframe
    Input : -directory_path : the path of the folder where the different runs considered are stored
            -hourly_read : boolean True if the runs include hourly data; False if not
            -name_folder_scenario : boolean True if the name of the different runs is scenario_k with k a natural number; False else
    Return : A dataframe where the columns are the variations of different relevant data 
    with the different run (which are the different lines)"""
    outputs = read_data_post_process(directory_path, hourly_read, name_folder_scenario)

    def add_list_storage (outputs, i, data):
        if ('energy_stored' not in outputs[i]):
            data['NG_STORAGE'].append(None)
            data['H2_STORAGE'].append(None)
            data['METHANOL_STORAGE'].append(None)
            data['AMMONIA_STORAGE'].append(None)
            data['WOOD_STORAGE'].append(None)
            data['MAX_STORAGE_GLOBAL'].append(None)
        else: 
            plotdata = outputs[i]['energy_stored']
            plotdata = plotdata.fillna(0)
            colonnes_zeros = plotdata.columns[plotdata.eq(0).all()]
            plotdata = plotdata.drop(colonnes_zeros, axis=1)
            col_sum = plotdata.abs().sum()

            # Supress vectors with not enough energy stored
            threshold = 8760
            cols_to_drop = col_sum[col_sum < threshold].index
            plotdata = plotdata.drop(cols_to_drop, axis=1)

            names_column = plotdata.columns
            plotdata = plotdata.reset_index()
            plotdata = plotdata[::24]
            column_storage_out = plotdata.columns[plotdata.columns.str.endswith('_out')]
            column_storage_in = plotdata.columns[plotdata.columns.str.endswith('_in')]
            names_column = names_column.difference(column_storage_in).difference(column_storage_out)
            plotdata[names_column] = plotdata[names_column] - plotdata[names_column].min(axis=0)
            ng_storage = max(plotdata['GAS_STORAGE'])
            h2_storage = max(plotdata['H2_STORAGE'])
            data['NG_STORAGE'].append(ng_storage)
            data['H2_STORAGE'].append(h2_storage)
            data['METHANOL_STORAGE'].append(max(plotdata['METHANOL_STORAGE']))
            data['AMMONIA_STORAGE'].append(max(plotdata['AMMONIA_STORAGE']))
            data['WOOD_STORAGE'].append(max(plotdata['WOOD_STORAGE']))
            data['MAX_STORAGE_GLOBAL'].append(max(plotdata.sum(axis=1)))
        return(data)

    
    data = {'CASE_NUMBER': [], 'FOLDER_NAME': [], 'SHARE_E_FUEL': [], 'SHARE_E_BIO_FUEL': [], 'SHARE_BIOFUEL': [], 'TOTAL_FUEL_PRODUCED': [],
                'COST': [], 'BIOMASS_SEQU': [], 'CO2_SEQUESTRATED': [], 'TOT_ELEC_PROD': [], 'HT_ELY_LOAD_FACTOR': [],
                'LIGNO_BIOMASS_PROD': [], 'CC_SHARE': [], 'QTT_JETFUEL_IMP': [], 'QTT_JETFUEL_RE_IMP': [],
                'ALK_ELY_LOAD_FACTOR': [], 'PV_PROD': [], 'TOT_WIND_PROD': [], 'H2_PROD': [],
                'NG_STORAGE': [], 'H2_STORAGE': [], 'METHANOL_STORAGE': [], 'AMMONIA_STORAGE': [],
                'CO2_DAC': [], 'METHANOL_PROD': [], 'TRUCK_H2': [], 'PRICE_QTT_RE_JETFUEL_IMPORT': [], 
                'SHARE_BIOFUEL_PROD_FT': [], 'SHARE_EBIOFUEL_PROD_FT': [], 'SHARE_EFUEL_PROD_FT': [],
                'HYDROGEN_PROD_FT': [], 'BIOMASS_PROD_FT': [],'CO2_PROD_FT': [], 'EMI_REDUC_JETFUEL': [], 'PRICE_JETFUEL_RE_IMPORT': [],
                'BIOMETHANE_PROD': [], 'E_BIOMETHANE_PROD': [], 'E_METHANE_PROD': [], 'PROD_ELEC_CCGT': [],
                'GASOLINE_TO_HVC': [], 'SHARE_METHANOL_TO_HVC': [], 'SHARE_JETFUEL_OUT_FT': [], 'WOOD_STORAGE': [], 'MAX_STORAGE_GLOBAL':[], 'BIOMASS_USE_DETAILED': [],
                'BIOMASS_USE_HEAT':[], 'F_PV': [], 'F_NUCLEAR': [], 'F_WIND_ONSHORE': [], 'F_WIND_OFFSHORE': [], 'F_E_WOOD_TO_FT': [], 'F_WOOD_TO_FT': [], 'F_CO2_TO_FT': [],
                'F_E_WOOD_TO_METHANOL': [], 'F_WOOD_TO_METHANOL': [], 'F_CO2_TO_METHANOL': [], 'F_HT_ELECTROLYSIS': [], 'F_ALKALINE_ELECTROLYSIS': [], 'F_BIOMASS_TO_HVC': [],
                'COST_VECTOR_ELEC': [], 'COST_VECTOR_JETFUEL': [], 'COST_VECTOR_H2': [], 'COST_VECTOR_BIOMASS': []}
    
    for i in range (0, len(outputs)):
        if outputs[i] == False:
            for key in data:
                if key != 'CASE_NUMBER':
                    data[key].append(None)
                else:
                    data[key].append(i)
        else:
            data['CASE_NUMBER'].append(i)
            data['FOLDER_NAME'].append(outputs[i]['folder_name'])
            # General data
            data['COST'].append(outputs[i]['cost_breakdown']['C_inv'].sum() + outputs[i]['cost_breakdown']['C_maint'].sum() + outputs[i]['cost_breakdown']['C_op'].sum())
            data['CO2_SEQUESTRATED'].append(-outputs[i]['year_balance'].loc['SEQUESTRATION','CO2_CAPTURED'])
            data['BIOMASS_SEQU'].append(outputs[i]['year_balance'].loc['BIOMASS_SEQUESTRATION','WOOD'])
            data['TOT_ELEC_PROD'].append(outputs[i]['year_balance'].loc[outputs[i]['year_balance'].loc[:,'ELECTRICITY' ].ge(100),'ELECTRICITY' ].sum())
            data['CC_SHARE'].append(100*(outputs[i]['year_balance'].loc['INDUSTRY_CCS','CO2_CAPTURED'])/( outputs[i]['year_balance'].loc[outputs[i]['year_balance'].loc[:,'CO2_CENTRALISED' ].ge(10),'CO2_CENTRALISED' ].sum()))
            data['LIGNO_BIOMASS_PROD'].append(outputs[i]['year_balance'].loc['WOOD_GROWTH','WOOD'])
            data['ALK_ELY_LOAD_FACTOR'].append(-outputs[i]['year_balance'].loc['ALKALINE_ELECTROLYSIS', 'ELECTRICITY']/(outputs[i]['assets'].loc['ALKALINE_ELECTROLYSIS', 'f']*8760))
            data['PV_PROD'].append(outputs[i]['year_balance'].loc['PV', 'ELECTRICITY'])
            data['TOT_WIND_PROD'].append(outputs[i]['year_balance'].loc['WIND_ONSHORE', 'ELECTRICITY'] + outputs[i]['year_balance'].loc['WIND_OFFSHORE', 'ELECTRICITY'])
            data['H2_PROD'].append(outputs[i]['year_balance'].loc['ALKALINE_ELECTROLYSIS', 'H2']+outputs[i]['year_balance'].loc['HT_ELECTROLYSIS', 'H2'])
            data['CO2_DAC'].append(outputs[i]['year_balance'].loc['DAC_LT', 'CO2_CAPTURED'] + outputs[i]['year_balance'].loc['DAC_HT', 'CO2_CAPTURED'])
            data['METHANOL_PROD'].append(outputs[i]['year_balance'].loc['CO2_TO_METHANOL', 'METHANOL'] + outputs[i]['year_balance'].loc['WOOD_TO_METHANOL', 'METHANOL'] + outputs[i]['year_balance'].loc['E_WOOD_TO_METHANOL', 'METHANOL'])
            data['SHARE_TRUCK_H2'].append((outputs[i]['year_balance'].loc['TRUCK_FUEL_CELL', 'MOB_FREIGHT_ROAD'])/(outputs[i]['year_balance'].loc[outputs[i]['year_balance'].loc[:,'MOB_FREIGHT_ROAD' ].ge(10),'MOB_FREIGHT_ROAD' ].sum()))
            # Data related to fuels 
            data['PRICE_QTT_RE_JETFUEL_IMPORT'].append(outputs[i]['cost_breakdown'].loc['JETFUEL_RE', 'C_op'])
            total_fuel_produced = outputs[i]['year_balance'].loc['CO2_TO_FT', 'FT_FUEL'] + outputs[i]['year_balance'].loc['CO2_TO_METHANOL', 'METHANOL'] + outputs[i]['year_balance'].loc['CO2_TO_METHANE', 'GAS'] + outputs[i]['year_balance'].loc['E_WOOD_TO_FT', 'FT_FUEL'] + outputs[i]['year_balance'].loc['E_WOOD_TO_METHANOL', 'METHANOL'] + outputs[i]['year_balance'].loc['E_WOOD_TO_METHANE', 'GAS'] + outputs[i]['year_balance'].loc['WOOD_TO_FT', 'FT_FUEL'] + outputs[i]['year_balance'].loc['WOOD_TO_METHANOL', 'METHANOL'] + outputs[i]['year_balance'].loc['WOOD_TO_METHANE', 'GAS']
            share_e_fuel = 100*(outputs[i]['year_balance'].loc['CO2_TO_FT', 'FT_FUEL'] + outputs[i]['year_balance'].loc['CO2_TO_METHANOL', 'METHANOL'] + outputs[i]['year_balance'].loc['CO2_TO_METHANE', 'GAS'])/total_fuel_produced
            share_e_bio_fuel = 100*(outputs[i]['year_balance'].loc['E_WOOD_TO_FT', 'FT_FUEL'] + outputs[i]['year_balance'].loc['E_WOOD_TO_METHANOL', 'METHANOL'] + outputs[i]['year_balance'].loc['E_WOOD_TO_METHANE', 'GAS'])/total_fuel_produced
            share_bio_fuel = 100*(outputs[i]['year_balance'].loc['WOOD_TO_FT', 'FT_FUEL'] + outputs[i]['year_balance'].loc['WOOD_TO_METHANOL', 'METHANOL'] + outputs[i]['year_balance'].loc['WOOD_TO_METHANE', 'GAS'])/total_fuel_produced
            data['SHARE_E_FUEL'].append(share_e_fuel)
            data['SHARE_E_BIO_FUEL'].append(share_e_bio_fuel)
            data['SHARE_BIOFUEL'].append(share_bio_fuel)
            data['TOTAL_FUEL_PRODUCED'].append(total_fuel_produced)
            data['QTT_JETFUEL_RE_IMP'].append(outputs[i]['year_balance'].loc['JETFUEL_RE','JETFUEL'])
            data['QTT_JETFUEL_IMP'].append(outputs[i]['year_balance'].loc['JETFUEL','JETFUEL'])
            data['PRICE_JETFUEL_RE_IMPORT'].append(outputs[i]['cost_breakdown'].loc['JETFUEL_RE', 'C_op']/outputs[i]['year_balance'].loc['JETFUEL_RE', 'JETFUEL'])
            data['SHARE_BIOFUEL_PROD_FT'].append(100*round(outputs[i]['year_balance'].loc['WOOD_TO_FT', 'FT_FUEL']/(outputs[i]['year_balance'].loc['CO2_TO_FT', 'FT_FUEL']+outputs[i]['year_balance'].loc['WOOD_TO_FT', 'FT_FUEL']+outputs[i]['year_balance'].loc['E_WOOD_TO_FT', 'FT_FUEL']), 3))
            data['SHARE_EBIOFUEL_PROD_FT'].append(100*round(outputs[i]['year_balance'].loc['E_WOOD_TO_FT', 'FT_FUEL']/(outputs[i]['year_balance'].loc['CO2_TO_FT', 'FT_FUEL']+outputs[i]['year_balance'].loc['WOOD_TO_FT', 'FT_FUEL']+outputs[i]['year_balance'].loc['E_WOOD_TO_FT', 'FT_FUEL']), 3))
            data['SHARE_EFUEL_PROD_FT'].append(100*round(outputs[i]['year_balance'].loc['CO2_TO_FT', 'FT_FUEL']/((outputs[i]['year_balance'].loc['CO2_TO_FT', 'FT_FUEL']+outputs[i]['year_balance'].loc['WOOD_TO_FT', 'FT_FUEL']+outputs[i]['year_balance'].loc['E_WOOD_TO_FT', 'FT_FUEL'])), 3))
            data['HYDROGEN_PROD_FT'].append(-round((outputs[i]['year_balance'].loc['CO2_TO_FT', 'H2']+outputs[i]['year_balance'].loc['WOOD_TO_FT', 'H2']+outputs[i]['year_balance'].loc['E_WOOD_TO_FT', 'H2']), 1))
            data['BIOMASS_PROD_FT'].append(-round((outputs[i]['year_balance'].loc['CO2_TO_FT', 'WOOD']+outputs[i]['year_balance'].loc['WOOD_TO_FT', 'WOOD']+outputs[i]['year_balance'].loc['E_WOOD_TO_FT', 'WOOD']), 1))
            data['CO2_PROD_FT'].append(-round((outputs[i]['year_balance'].loc['CO2_TO_FT', 'CO2_CAPTURED']+outputs[i]['year_balance'].loc['WOOD_TO_FT', 'CO2_CAPTURED']+outputs[i]['year_balance'].loc['E_WOOD_TO_FT', 'CO2_CAPTURED']), 1))
            data['EMI_REDUC_JETFUEL'].append(0)#str(int(0.1-outputs[i]['year_balance'].loc['JETFUEL_RE', 'CO2_ATMOSPHERE']/outputs[i]['year_balance'].loc['JETFUEL_RE', 'JETFUEL']*100/0.26))) #Return error if there is no import of renewable jet fuel
            data['BIOMETHANE_PROD'].append(outputs[i]['year_balance'].loc['WOOD_TO_METHANE', 'GAS']/1000)
            data['E_BIOMETHANE_PROD'].append(outputs[i]['year_balance'].loc['E_WOOD_TO_METHANE', 'GAS']/1000)
            data['E_METHANE_PROD'].append(outputs[i]['year_balance'].loc['CO2_TO_METHANE', 'GAS']/1000)
            data['GASOLINE_TO_HVC'].append(outputs[i]['year_balance'].loc['OIL_TO_HVC', 'HVC'])
            data['SHARE_METHANOL_TO_HVC'].append(100*outputs[i]['year_balance'].loc['METHANOL_TO_HVC', 'HVC']/outputs[i]['year_balance'].loc[outputs[i]['year_balance'].loc[:,'HVC' ].ge(100),'HVC' ].sum())
            data['SHARE_JETFUEL_OUT_FT'].append(-outputs[i]['year_balance'].loc['REFINERY_JETFUEL', 'JETFUEL']/outputs[i]['year_balance'].loc['REFINERY_JETFUEL', 'FT_FUEL'])
            data['PROD_ELEC_CCGT'].append(outputs[i]['year_balance'].loc['CCGT', 'ELECTRICITY'])
            data['HT_ELY_LOAD_FACTOR'].append(-outputs[i]['year_balance'].loc['HT_ELECTROLYSIS', 'ELECTRICITY']/(outputs[i]['assets'].loc['HT_ELECTROLYSIS', 'f']*8760))
            data['BIOMASS_USE_HEAT'].append(-(outputs[i]['year_balance'].loc['IND_BOILER_WOOD', 'WOOD'] + outputs[i]['year_balance'].loc['DEC_BOILER_WOOD', 'WOOD'] + outputs[i]['year_balance'].loc['DHN_BOILER_WOOD', 'WOOD'])/1000)
            data['BIOMASS_USE_DETAILED'].append(outputs[i]['year_balance']['WOOD'])
            # Data related to capacity installed
            data['F_PV'].append(outputs[i]['assets'].loc['PV', 'f'])
            data['F_NUCLEAR'].append(outputs[i]['assets'].loc['NUCLEAR', 'f'])
            data['F_WIND_ONSHORE'].append(outputs[i]['assets'].loc['WIND_ONSHORE', 'f'])
            data['F_WIND_OFFSHORE'].append(outputs[i]['assets'].loc['WIND_OFFSHORE', 'f'])
            data['F_E_WOOD_TO_FT'].append(outputs[i]['assets'].loc['E_WOOD_TO_FT', 'f'])
            data['F_WOOD_TO_FT'].append(outputs[i]['assets'].loc['WOOD_TO_FT', 'f'])
            data['F_CO2_TO_FT'].append(outputs[i]['assets'].loc['CO2_TO_FT', 'f'])
            data['F_E_WOOD_TO_METHANOL'].append(outputs[i]['assets'].loc['E_WOOD_TO_METHANOL', 'f'])
            data['F_WOOD_TO_METHANOL'].append(outputs[i]['assets'].loc['WOOD_TO_METHANOL', 'f'])
            data['F_CO2_TO_METHANOL'].append(outputs[i]['assets'].loc['CO2_TO_METHANOL', 'f'])
            data['F_ALKALINE_ELECTROLYSIS'].append(outputs[i]['assets'].loc['ALKALINE_ELECTROLYSIS', 'f'])
            data['F_HT_ELECTROLYSIS'].append(outputs[i]['assets'].loc['HT_ELECTROLYSIS', 'f'])
            data['F_BIOMASS_TO_HVC'].append(outputs[i]['assets'].loc['BIOMASS_TO_HVC', 'f'])
            add_list_storage (outputs, i, data)
            
    df = pd.DataFrame(data)
    return(df)