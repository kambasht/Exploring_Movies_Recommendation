# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import operator
import seaborn as sns #seaborn plot for visualization
sns.set(style="darkgrid")
import matplotlib as mpl
import matplotlib.pyplot as plt

def explore_df(df):
    '''
    INPUT
    df - A dataframe
    
    OUTPUT
    print values of different exploratory data analysis steps on the output console
    
    This function :
    1. prints the shape of the dataframe
    2. returns a count and list of categorical and numerical columns
    3. uses the describe() method to print statistics for the numerical columns
    '''
    
    # display the shape, gives us number of rows and column
    print("The dataframe has ",df.shape[0],"rows and ", df.shape[1],"columns \n")
    
    # Subset to a dataframe only holding the categorical columns
    cat_df = return_dtype(df,"object")
    
    # Subset to a dataframe only holding the numerical columns
    num_df = return_dtype(df,"number")
    
    print("There are ",cat_df.shape[1],"categorical columns and ",num_df.shape[1],"numerical columns in the dataframe \n")
    
    print("A list of all the categorical columns \n")
    print(list(cat_df.columns),"\n")
    
    for col in list(cat_df.columns):
        #df[col] = df[col].astype(str)
        try:
            if len(list(np.unique(df[col]))) <= 10:
                print("Unique values in column ",col,":",list(np.unique(df[col])),"\n")
            else:
                print("Total number of unique values for column ",col,"is ",len(list(np.unique(df[col]))),"\n\nThe first 10 unique values are", list(np.unique(df[col]))[:10])
  
        except:
            pass
    print("\n A list of all the numerical columns \n")
    print(list(num_df.columns),"\n")
    
    # gives us a snapshot of all numerical columns and related statistics like mean, min and max values
    print("Statistical details for the numerical columns \n",df.describe())
    
def return_dtype(df,dtype):
    '''
    INPUT
    df - A dataframe
    d_type - the data type of the columns you need to pull from the dataframe i.e. 'number','object'
    
    OUTPUT
    dataframe with only columns of 'dtype' datatype
    
    This function :
    returns a dataframe with columns of the specified data types
    '''
    return df.select_dtypes(include=[dtype]).copy()



def explore_null(df):
    '''
    INPUT
    df - A dataframe
    
    OUTPUT
    print values of columns which have 25,50 or 75% null values in them
    
    This function :
    1. prints the column which has all null values
    2. print values of columns which have 25,50 or 75% null values in them
    '''
    
    # print list of columns that have "all" null values
    all_nulls = set(df.columns[df.isnull().all()])
    print("The column(s) with all null values: ",all_nulls,"\n")
    
    for perc in range(25,100,25):
        # columns with more than x% of null values
        print(" More than ",perc,"% of values are null for columns ",set(df.columns[df.isnull().mean() > (perc/100)]),"\n")
        
def drop_all_nulls(df):
    '''
    INPUT
    df - A dataframe
    
    OUTPUT
    df - A modified dataframe with any row or column with 'all' null values dropped
    
    This function :
    any row or column with all null values are dropped
    '''
    
    # drop any columns (axis=1) which have 'all' null values
    # make sure you explicitly mention the axis as 1 to indicate you are removing columns
    # default for dropna method is axis=0 which means any rows with null values will be removed
    df = df.dropna(how="all", axis=1)
    
    # drop any rows (axis=0) which have all null values
    df = df.dropna(how="all", axis=0)
    
    return df

def one_hot_encode(df):
    '''
    INPUT
    df - A dataframe
    
    OUTPUT
    df - modified dataframe
    
    This function :
    1. encodes all categorical columns with one hot encoding i.e. each value in the column is separated out into
       a different column, column names separated with '_'
    2. drops all original categorical columns
    
    Reason behind one hot encoding: 
    Categorical columns do not fit into a linear regression model. One-hot encoding is a great tool for turning some of 
    these categorical features into multiple binary features; the presence or absence of the individual categorical unit 
    can then be fit into the linear regression.
    
    (Credits: https://medium.com/@jjosephmorrison/one-hot-encoding-to-set-up-categorical-features-for-linear-regression-6bac35661bb6)
    '''
    # return all categorical columns
    cat_df = return_dtype(df,'object')
    # store the column list
    cat_cols = cat_df.columns

    # for each categorical column add dummy var, drop original column
    for col in cat_cols:
        try:

            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=False, dummy_na=False)], axis=1)
        except:
            continue
    return df

def one_hot_encode_for_lists(df,col):
    '''
    INPUT
    df - A dataframe
    col - column which contains values as lists
    
    OUTPUT
    df - modified dataframe
    
    This function :
    1. is specifically built to one hot encode columns in a dataframe where values are lists
    2. encodes the provided column with one hot encoding i.e. each value in the column is separated out into
       a different column, the list values become new column names
    3. drops all original categorical columns
    
    Reason behind one hot encoding: 
    Categorical columns do not fit into a linear regression model. One-hot encoding is a great tool for turning some of 
    these categorical features into multiple binary features; the presence or absence of the individual categorical unit 
    can then be fit into the linear regression.
    
    (Credits: https://medium.com/@jjosephmorrison/one-hot-encoding-to-set-up-categorical-features-for-linear-regression-6bac35661bb6)
    '''
    # initialize empty set to store item values
    temp_set = set()
    # iterate through all the values in columns
    for item in df[col].values:
        for val in list(item):
            # add value to set
            temp_set.add(val)
            
    # parse through each row of dataframe
    for i in range(df.shape[0]):
        # for each value in the set created above
        for val in temp_set:
            # check if that value exists in the element list
            if val in (df.iloc[i,df.columns.get_loc(col)]):
                # store 1 if it exists
                df.loc[i,val]=int(1)
            else:
                df.loc[i,val]=int(0)
    
    # drop original column
    df = df.drop([col],axis=1)
    
    # return modified dataframe
    return df

def map_ids(df, col):
    '''
    INPUT
    df - A dataframe
    col - column of the dataframe that needs to be mapped
    
    OUTPUT
    id_dict -a dictionary with the new numerical values i.e. 0,1,2 as values and the older strings as keys
    
    This function :
    allows for easy mapping of hashed/crypted values like emails and names to numerical values for easier
    data joins and munging
    
    '''
    #initiate empty dictionary
    id_dict = dict()
    #initialize a counter value
    ctr = 0 
    # for each value in the column
    for val in list(df[col].values):
        # if we encounter a new value that becomes our dict key
        if val not in id_dict.items():
            # the ctr values becomes our dict value
            id_dict[val] = ctr
            ctr+=1
  
    # return the dictionary
    return id_dict


def find_similar_offers(offer_id, portfolio):
    '''
    INPUT:
    offer_id - (int) a offer_id in portofolio 
    
    OUTPUT:
    most_similar_offer - the most similar offer_id with the one which was input
    
    Description:
    Computes the similarity of every pair of offer based on pearson's coefficient
    
    '''
    try: 
        # create similar offers dictionary, we will be using this to store the most similar offer and 
        # their corresponding pearson coefficient in a key,value pair
        similar_offers = dict()
        
        # we must remove the offer id column to avoid any correlation within the offer id values themselves
        portfolio = portfolio.drop('offer_id',axis=1)
        
        # compute similarity of each offer to the provided offer
        # move through every offer in portfolio
        for i in range(portfolio.shape[0]):
            similar_offers[i] = np.corrcoef(portfolio.iloc[offer_id,], portfolio.iloc[i,])[0][1]

        # sort by similarity
        sorted_similar_offers = sorted(similar_offers.items(), key=operator.itemgetter(1),reverse=True)
        # pop the first one because it will always be the offer itself
        sorted_similar_offers.pop(0)
        # pick the offer id
        most_similar_offer = sorted_similar_offers[0][0]
        
        #most_similar_offer = portfolio.query('offer_id == @most_similar_offer')
    
    except:
        print("offer not found in portfolio")
        return
       
    return most_similar_offer # return a list of the users in order from most to least similar

def corr_plot(df):
    '''
    INPUT:
    df : a dataframe
    
    OUTPUT:
    a seaborn correlation plot
    
    Description:
    Helps plot the correlation matrix for a dataframe
    '''
    
    # The following piece of code has been borrowed from the official seaborn website, example for pairwise correlation
    # Credits: https://seaborn.pydata.org/examples/many_pairwise_correlations.html

    sns.set(style="white",font_scale=1.5)

    # Generate a large random dataset
    rs = np.random.RandomState(33)
    d = pd.DataFrame(data=rs.normal(size=(100, df.shape[1])),
                     columns=list(df.columns))

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 16))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(1000,8, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.8, cbar_kws={"shrink": 0.8});

    # Visual changes for the map
    plt.xticks(rotation=90)

    # set x and y text labels
    plt.suptitle("Correlation matrix", fontsize=20)

    plt.show()
    
def retrieve_tag_dataframe(df,col_name,tag_to_search,pos_from_tag,split_on,row_num, first_n_values=1):
    '''
    INPUT:
    df : dataframe to be searched
    col_name: the column to be searched
    tag_to_search: the tag to be searched for i.e. "\name\:"
    pos_from_tag: the number of positions from the tag, where the value that needs to be extracted starts i.e. 3
    split_on: once extracted, what to split on the string to pick the first piece i.e. can be "," or "}"
    row_num : the row number to be searched
    first_n_values: how many occurences of the tag do we need to pull i.e. 4
    
    OUTPUT:
    list with the cleaned up values as elements
    
    Description:
    on the string provided (x), finds the tag, picks up the value at the "pos_from_tag" location, splits and picks the first half
    
    '''
    name_pos_list= [m.start()+pos_from_tag for m in re.finditer(tag_to_search, df[col_name][row_num])]
    j=0
    val_list=list()
    
    # iterate through the positions and pick genres
    for i in name_pos_list:
        if j < first_n_values:
            val_list.append(df[col_name][row_num][i:].split(split_on)[0].replace("'","").replace(" ","").strip(""))
            j+=1
    return val_list



def retrieve_tag_field(x,tag_to_search,pos_from_tag,split_on, first_n_values=1):
    '''
    INPUT:
    x : the string to be searched for i.e. "\name\:'John Doe'"
    tag_to_search: the tag to be searched for i.e. "\name\:"
    pos_from_tag: the number of positions from the tag, where the value that needs to be extracted starts i.e. 3
    split_on: once extracted, what to split on the string to pick the first piece i.e. can be "," or "}"
    first_n_values: how many occurences of the tag do we need to pull i.e. 4
    
    OUTPUT:
    list with the cleaned up values as elements
    
    Description:
    on the string provided (x), finds the tag, picks up the value at the "pos_from_tag" location, splits and picks the first half
    
    '''
    # the finditer allows us to store a list of all the positions of the value we are looking for
    name_pos_list= [m.start()+pos_from_tag for m in re.finditer(tag_to_search, x)]
    j=0
    # to store the values, will be returned as the final output
    val_list=list()
    
    # iterate through the positions and pick genres
    for i in name_pos_list:
        # only loop through the first_n_values provided by user, 1 by default
        if j < first_n_values:
            val_list.append(x[i:].split(split_on)[0].replace("'","").replace(" ","").strip(""))
            j+=1
    return val_list
