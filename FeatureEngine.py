from numpy import mean, absolute
import pandas as pd
import re, os
from pymatgen.core.composition import Composition

class Features:
    def __init__(self,formula_file):
        self.formula_file=formula_file
        self.atomic_data_file='DATA/atomic_data.csv'

    def make_features(self,atomic_descriptors,formula_list,targets,addAVG=True,addAAD=True,addMD=False,addCV=False):
        print('--------Generating Features------------')
        all_feature_list=[]
        for indx0, formula in enumerate(formula_list):
            #print(f'{indx0}/{len(formula_list)} -> formula: {formula}')
            feature_list = []
            atom_symbols=list(atomic_descriptors[0].keys())

            comp = Composition(formula)
            formula=comp.formula
            s = re.findall('([A-Z][a-z]?)([0-9]?\.?[0-9]*)', formula)
            comp_vector = [0 for x in range(0, len(atom_symbols))]

            feature_list.append(targets[indx0])

            # Calculating the total number of atoms in the chemical formula
            total = 0
            for elem, num in s:
                if (num == ''):
                    num = 1
                total += int(num)

            # Calculating Weighted Average
            avg = 0
            for des in atomic_descriptors:
                des_list = []
                for elem, num in s:
                    if (num == ''):
                        num = 1
                    num = int(num)
                    avg += des[elem] * num
                    des_list.append((des[elem],num))
                avg = avg / total
                if(addAVG):
                    feature_list.append(avg)

                # Calculating Average Absolute Deviation
                if (addAAD):
                    avgAD = 0
                    for y, num in des_list:
                        ad = abs(y - avg)*num
                        avgAD += ad
                    avgAD = avgAD /total


                    feature_list.append(avgAD)

                # Calculating maximum difference
                if (addMD):
                    dif_list=[]
                    for y1, num1 in des_list:
                        for y2, num2 in des_list:
                            dif=abs(y1-y2)
                            dif_list.append(dif)
                    max_dif=max(dif_list)

                    feature_list.append(max_dif)

            # Creating Element Ratio Vector
            for elem, num in s:
                if (num == ''):
                    num = 1
                num = int(num)
                index = atom_symbols.index(elem)
                comp_vector[index] = int(num) / total

            # Uncomment if the element ratio vector is required
            if(addCV):
                for ratio in comp_vector:
                    feature_list.append(ratio)

            all_feature_list.append(feature_list)

        return all_feature_list

    def get_formula_list(self):
        df_mat = pd.read_csv(self.formula_file, header=None)
        formula_list = [x[0] for x in df_mat.values.tolist()]

        return formula_list

    def get_encoded_sym(self):
        df_mat = pd.read_csv(self.formula_file, header=None)
        sym_list = [x[2] for x in df_mat.values.tolist()]
        encoded_sym=[]
        for sym in sym_list:
            if(sym=='monoclinic'):
                digits = [1,0,0,0,0,0,0]
            elif (sym == 'triclinic'):
                digits = [0, 1, 0, 0, 0, 0, 0]
            elif (sym == 'orthorhombic'):
                digits = [0, 0, 1, 0, 0, 0, 0]
            elif (sym == 'trigonal'):
                digits = [0, 0, 0, 1, 0, 0, 0]
            elif (sym == 'hexagonal'):
                digits = [0, 0, 0, 0, 1, 0, 0]
            elif (sym == 'cubic'):
                digits = [0, 0, 0, 0, 0, 1, 0]
            elif (sym == 'tetragonal'):
                digits = [0, 0, 0, 0, 0, 0, 1]

            encoded_sym.append(digits)
        return encoded_sym

    def get_targets(self):
        df_mat = pd.read_csv(self.formula_file, header=None)
        targets = [x[1] for x in df_mat.values.tolist()]

        return targets

    def get_atomic_descriptors(self):
        df_des = pd.read_csv(self.atomic_data_file, header=None)

        atomic_descriptors=[]

        elements = [x[0] for x in df_des.values.tolist()]
        for i in range(1, len(df_des.columns)):
            tmp = [x[i] for x in df_des.values.tolist()]
            des_dict = dict(zip(elements, tmp))

            atomic_descriptors.append(des_dict)

        return atomic_descriptors

    def get_features(self,addAVG=True,addAAD=True,addMD=False,addCV=False):
        formula_list = self.get_formula_list()
        #encoded_sym=self.get_encoded_sym()
        targets=self.get_targets()
        atomic_descriptors=self.get_atomic_descriptors()

        features = self.make_features(atomic_descriptors=atomic_descriptors,
                                      targets=targets,formula_list=formula_list,addAVG=addAVG,addAAD=addAAD,addMD=addMD,addCV=addCV)

        return features