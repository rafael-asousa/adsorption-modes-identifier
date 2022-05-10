import os
import abc
import sys
import pandas as pd

from glob               import glob
from molecule_functions import *

from sklearn.decomposition     import PCA
from sklearn.preprocessing     import StandardScaler
from yellowbrick.cluster.elbow import KElbowVisualizer
from sklearn.cluster           import KMeans

class AdsorptionModesIdentifier():
    def __init__(self, data_collector, data_transformer, data_classifier) -> None:
        self.collector = data_collector
        self.transformer = data_transformer
        self.classifier = data_classifier

    def identify_modes(self):
        original_data = self.collector.obtain_data()
        data = self.transformer.transform_data(original_data)
        labels = self.classifier.modes_classifier(data)

        original_data['labels'] = labels

        original_data.to_csv('./clustered_data.csv')

        return None


class DataCollector(abc.ABC):
    def __init__(self, path = os.getcwd()):
        if os.path.exists(path):
            self.path = path
        else:
            raise IOError('path for structures does not exist')
    
    def _get_configuration_number(self, filename):
        filename = filename.split('/')[-1]
        conf_num = filename.split('_')[-1].split('.')[0]
        
        return conf_num

    @abc.abstractmethod
    def _get_molecule_data():
        pass

    @abc.abstractmethod
    def _get_columns_names():
        pass

    def obtain_data(self) -> pd.DataFrame:
        
        df_cols = self._get_columns_names().insert(0, 'configuration_number')
        df_structures_data = pd.DataFrame(columns=df_cols)

        for file in glob(self.path+'/*.xyz'):
            conf_num = self._get_configuration_number(file)
            structure_data = self._get_molecule_data(file)
            structure_data['configuration_number'] = conf_num

            df_structures_data = df_structures_data.append(structure_data, ignore_index=True)

        return df_structures_data


class DataTransformer():
    def __init__(self, embedding=PCA(), scaler=StandardScaler()):
        self.embedding = embedding
        self.scaler = scaler

    def _categorical_columns(self, data: pd.DataFrame) -> list:
        return data.select_dtypes(include=['object']).columns

    def _numerical_columns(self, data: pd.DataFrame) -> list:
        return data.select_dtypes(exclude=['object']).columns
          
    def _apply_scaling(self, data:pd.DataFrame):
        scaled_df = pd.DataFrame(self.scaler.fit_transform(data), columns = data.columns)
        return scaled_df

    def _apply_embedding(self, data: pd.DataFrame):
        return pd.DataFrame(self.embedding.fit_transform(data))

    def transform_data(self, data: pd.DataFrame):
        data = data.drop(['configuration_number','energy'], axis=1)

        categorical_data = data[self._categorical_columns(data)]
        numerical_data   = data[self._numerical_columns(data)]
        
        numerical_data = self._apply_scaling(numerical_data)
        categorical_data = pd.get_dummies(categorical_data)

        data = pd.concat([numerical_data, categorical_data], axis=1)

        data = self._apply_embedding(data)

        return data

class DataClassifier():
    def __init__(self, k:int = None):
        self.k = k
    
    def _choose_K(self, scaled_data: pd.DataFrame, metric: str = 'distortion'):
        visualizer = KElbowVisualizer(KMeans(random_state=42), k=(3,10), metric=metric).fit(scaled_data)
        k = visualizer.elbow_value_

        return k

    def modes_classifier(self, scaled_data: pd.DataFrame) -> pd.DataFrame:
        if self.k == None:
            self.k = self._choose_K(scaled_data)

        kmeans = KMeans(n_clusters=self.k, random_state=42)
        kmeans.fit(scaled_data)
        
        scaled_data['labels'] = kmeans.labels_
        
        return scaled_data['labels']


class CO2_collector(DataCollector):
    def __init__(self, path):
        super().__init__(path=path)
        self.C=0
        self.O1=1
        self.O2=2
        self.n_atoms = 3

    def _get_molecule_data(self, filename: str):
        energy = GetEnergy(filename)
            
        #Get the minimal dists of each molecule atom and to which atom it is closest in the cluster
        _, _, minimal_dists, atoms = MolMinDists(filename, self.n_atoms)

        #Get the structural properties of molecule
        OCO_ang = BondAngle(self.O1, self.C, self.O2, filename)
        dCO1 = BondDist(self.C, self.O1, filename)
        dCO2 = BondDist(self.C, self.O2, filename)
        
        bond_lengths, close_atom, far_atom = OrderAtomsByDist(dCO1, self.O1, dCO2, self.O2)

        df_conf = pd.DataFrame({'dC_cluster': minimal_dists[0],
                                'dO1_cluster': minimal_dists[close_atom],
                                'dO2_cluster': minimal_dists[far_atom],
                                'C_bond':closest_cluster_atom(self.C, atoms),
                                'closeO_bond':closest_cluster_atom(close_atom, atoms),
                                'farO_bond':closest_cluster_atom(far_atom, atoms),
                                'OCO_angle': OCO_ang,
                                'biggest_CO_length': bond_lengths[1],
                                'smallest_CO_length': bond_lengths[0],
                                'energy':energy}, index=[0])

        return df_conf
    
    def _get_columns_names(self) -> list:
        return ['dC_cluster',
                'dO1_cluster',
                'dO2_cluster',
                'C_bond',
                'closeO_bond',
                'farO_bond',
                'OCO_angle',
                'biggest_CO_length',
                'smallest_CO_length',
                'energy']

class CO_collector(DataCollector):
    def __init__(self, path):
        super().__init__(path=path)
        self.C = 0
        self.O = 1
        self.n_atoms = 2

    def _get_molecule_data(self, filename: str):            
        _, _, minimal_dists, atoms = MolMinDists(filename, self.n_atoms)

        dCO = BondDist(self.C, self.O, filename)
        
        df_conf = pd.DataFrame({'dC_cluster': minimal_dists[self.C],
                                'dO_cluster': minimal_dists[self.O],
                                'C_bond':closest_cluster_atom(self.C, atoms),
                                'O_bond':closest_cluster_atom(self.O, atoms),
                                'CO_length': dCO}, index=[0])

        return df_conf
    
    def _get_columns_names(self) -> list:
        return ['dC_cluster',
                'dO_cluster',
                'C_bond',
                'O_bond',
                'CO_length']

class H2_collector(DataCollector):
    def __init__(self, path):
        super().__init__(path=path)
        self.H1 = 0
        self.H2 = 1
        self.n_atoms = 2

    def _get_molecule_data(self, filename: str):            
        _, _, minimal_dists, atoms = MolMinDists(filename)
                
        dHH = BondDist(self.H1, self.H2, filename)
        
        dH1Cluster = minimal_dists[0]
        dH2Cluster = minimal_dists[1]
        
        if dH1Cluster < dH2Cluster:
            close_H = self.H1
            far_H = self.H2
        else:
            close_H = self.H2
            far_H = self.H1
               
        df_conf = pd.DataFrame({'dH1_cluster': minimal_dists[far_H],
                                'dH2_cluster':minimal_dists[close_H],
                                'H1_bond':atoms[close_H][0], 
                                'H2_bond':atoms[far_H][0],
                                'HH_length':  dHH}, index=[0])

        return df_conf
    
    def _get_columns_names(self) -> list:
        return ['dH1_cluster',
                'dH2_cluster',
                'H1_bond',
                'H2_bond'
                'HH_length']


class HCOO_collector(DataCollector):
    def __init__(self, path=os.getcwd()):
        super().__init__(path)
        self.C=0
        self.O1=1
        self.O2=2
        self.H=3

    def _get_molecule_data(self, filename: str):
        _, _, minimal_dists, atoms = MolMinDists(filename)
                
        dO1 = minimal_dists[self.O1]
        dO2 = minimal_dists[self.O2]
        
        bond_lengths, close_O, far_O = OrderAtomsByDist(dO1, self.O1, dO2, self.O2)
        
        OCO_ang = BondAngle(self.O1, self.C, self.O2, filename)
        HCO_ang = BondAngle(self.H, self.C, close_O, filename)
        dCH = BondDist(self.C, self.H, filename)
        dCfar_O = BondDist(self.C, far_O, filename)
        dCclose_O = BondDist(self.C, close_O, filename)

        df_conf = pd.DataFrame({'dC_cluster': minimal_dists[self.C],
                                'dO1_cluster':minimal_dists[far_O], 
                                'dO2_cluster':minimal_dists[close_O], 
                                'dH_cluster':minimal_dists[self.H],
                                'C_bond':atoms[self.C][0], 
                                'O1_bond':atoms[far_O][0], 
                                'O2_bond':atoms[close_O][0], 
                                'H_bond':atoms[self.H][0], 
                                'OCO_angle':OCO_ang, 
                                'HCO_angle':HCO_ang, 
                                'CH_length':dCH, 
                                'CO1_length':dCfar_O, 
                                'CO2_length':dCclose_O}, index=[0])
        
        return df_conf
    
    def _get_columns_names(self) -> list:
        return ['dC_cluster',
                'dO1_cluster',
                'dO2_cluster',
                'dH_cluster',
                'C_bond',
                'O1_bond',
                'O2_bond',
                'H_bond',
                'OCO_angle',
                'HCO_angle',
                'CH_length',
                'CO1_length',
                'CO2_length']

class COOH_collector(DataCollector):
    def __init__(self, path=os.getcwd()):
        super().__init__(path)
        self.C=0
        self.O1=1
        self.O2=2
        self.H=3

    def _get_molecule_data(self, filename:str):    
        _, _, minimal_dists, atoms = MolMinDists(filename)
                
        O1CO2_ang = BondAngle(self.O1, self.C, self.O2, filename)
        CO2H_ang = BondAngle(self.C, self.O2, self.H, filename)
        dCO1 = BondDist(self.C, self.O1, filename)
        dCO2 = BondDist(self.C, self.O2, filename)
        dO2H = BondDist(self.O2, self.H, filename)

        df_conf = pd.DataFrame({'dC_cluster':minimal_dists[self.C],
                                'dO1_cluster':minimal_dists[self.O1],
                                'dO2_cluster':minimal_dists[self.O2],
                                'dH_cluster':minimal_dists[self.H],
                                'C_bond':atoms[self.C][0],
                                'O1_bond':atoms[self.O1][0],
                                'O2_bond':atoms[self.O2][0],
                                'H_bond':atoms[self.H][0],
                                'OCO_angle':O1CO2_ang,
                                'COH_angle':CO2H_ang,
                                'CO1_length':dCO1,
                                'CO2_length':dCO2,
                                'O2H_length':dO2H}, index=[0])

        return df_conf

    def _get_columns_names(self):
        return ['dC_cluster',
                'dO1_cluster',
                'dO2_cluster',
                'dH_cluster',
                'C_bond',
                'O1_bond',
                'O2_bond',
                'H_bond',
                'OCO_angle',
                'COH_angle',
                'CO1_length',
                'CO2_length',
                'O2H_length']
        

class COH_collector(DataCollector):
    def __init__(self, path=os.getcwd()):
        super().__init__(path)
        self.C = 0
        self.O = 1
        self.H = 2

    def _get_molecule_data(self, filename: str):
        _, _, minimal_dists, atoms = MolMinDists(filename)
        
        dCO = BondDist(self.C, self.O, filename)
        dOH = BondDist(self.O, self.H, filename)
        COH_ang = BondAngle(self.C, self.O, self.H, filename)
        
        df_conf = pd.DataFrame({'dC_cluster':minimal_dists[self.C],
                                'dO_cluster':minimal_dists[self.O],
                                'dH_cluster':minimal_dists[self.H],
                                'C_bond':atoms[self.C][0],
                                'O_bond':atoms[self.O][0],
                                'H_bond':atoms[self.H][0],
                                'CO_length':dCO,
                                'OH_length':dOH,
                                'COH_angle':COH_ang}, index=[0])

        return df_conf
        
    def _get_columns_names(self) -> list:
        return ['dC_cluster',
                'dO_cluster',
                'dH_cluster',
                'C_bond',
                'O_bond',
                'H_bond',
                'CO_length',
                'OH_length',
                'COH_angle']

class HCO_collector(DataCollector):
    def __init__(self, path=os.getcwd()):
        super().__init__(path)
        self.C = 0
        self.H = 1
        self.O = 2

    def _get_molecule_data(self, filename: str):
        _, _, minimal_dists, atoms = MolMinDists(filename)
        
        dCO = BondDist(self.C, self.O, filename)
        dCH = BondDist(self.C, self.H, filename)
        HCO_ang = BondAngle(self.H, self.C, self.O, filename)
        
        df_conf = pd.DataFrame({'dC_cluster':minimal_dists[self.C],
                                'dO_cluster':minimal_dists[self.O],
                                'dH_cluster':minimal_dists[self.H],
                                'C_bond':atoms[self.C][0],
                                'O_bond':atoms[self.O][0],
                                'H_bond':atoms[self.H][0],
                                'CO_length':dCO,
                                'CH_length':dCH,
                                'HCO_angle':HCO_ang}, index=[0])

        return df_conf
    
    def _get_columns_names(self) -> list:
        return ['dC_cluster',
                'dO_cluster',
                'dH_cluster',
                'C_bond',
                'O_bond',
                'H_bond',
                'CO_length',
                'CH_length',
                'HCO_angle']

if __name__ == '__main__':

    mol_path=sys.argv[1]

    identifier = AdsorptionModesIdentifier( data_collector   = CO2_collector(path = mol_path),
                                            data_transformer = DataTransformer(),
                                            data_classifier  = DataClassifier())

    identifier.identify_modes()