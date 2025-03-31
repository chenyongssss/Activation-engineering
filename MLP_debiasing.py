import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import random
import pandas as pd
import statsmodels.api as smapi
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

###########################################
# 1. Functions for fitting the null distribution and computing systematic error
###########################################

def gaussian_product(mu1, mu2, sd1, sd2):
    """
    Compute the product of two Gaussian density functions.
    """
    sqr = lambda x: x**2
    denominator = np.sqrt(2 * np.pi) * np.sqrt(sqr(sd1) + sqr(sd2))
    exponent = -sqr(mu1 - mu2) / (2 * (sqr(sd1) + sqr(sd2)))
    return (1 / denominator) * np.exp(exponent)

def log_likelihood_null(theta, logRr, seLogRr):
    """
    Compute the negative log-likelihood for the null distribution.
    theta[0] is the mean; theta[1] is the precision (1/sd^2).
    """
    if theta[1] <= 0:
        return np.inf
    result = 0.0
    sd = 1 / np.sqrt(theta[1])
    if sd < 1e-6:
        for i in range(len(logRr)):
            result -= norm.logpdf(theta[0], loc=logRr[i], scale=seLogRr[i])
    else:
        for i in range(len(logRr)):
            gaussian_val = gaussian_product(logRr[i], theta[0], seLogRr[i], sd)
            if gaussian_val <= 0:
                return np.inf
            result -= np.log(gaussian_val)
    if result == 0:
        return np.inf
    return result

def fit_null(logRr, seLogRr):
    """
    Fit the null distribution using L-BFGS-B optimization.
    Returns a dictionary with fitted 'mean' and 'sd'.
    """
    logRr = np.array(logRr)
    seLogRr = np.array(seLogRr)
    
    # Clean infinite or NaN values
    if np.any(np.isinf(seLogRr)):
        warnings.warn("Infinite standard errors detected. Removing before fitting null distribution.")
        valid_indices = ~np.isinf(seLogRr)
        logRr = logRr[valid_indices]
        seLogRr = seLogRr[valid_indices]
    if np.any(np.isinf(logRr)):
        warnings.warn("Infinite logRr detected. Removing before fitting null distribution.")
        valid_indices = ~np.isinf(logRr)
        logRr = logRr[valid_indices]
        seLogRr = seLogRr[valid_indices]
    if np.any(np.isnan(seLogRr)):
        warnings.warn("NaN standard errors detected. Removing before fitting null distribution.")
        valid_indices = ~np.isnan(seLogRr)
        logRr = logRr[valid_indices]
        seLogRr = seLogRr[valid_indices]
    if np.any(np.isnan(logRr)):
        warnings.warn("NaN logRr detected. Removing before fitting null distribution.")
        valid_indices = ~np.isnan(logRr)
        logRr = logRr[valid_indices]
        seLogRr = seLogRr[valid_indices]
    
    if len(logRr) == 0:
        warnings.warn("No estimates remaining.")
        return {"mean": np.nan, "sd": np.nan}
    
    theta = np.array([0, 1])  # initial guess: mean=0, precision=1
    result = minimize(
        log_likelihood_null,
        theta,
        args=(logRr, seLogRr),
        method='L-BFGS-B',
        bounds=[(None, None), (1e-6, None)]
    )
    
    if result.success:
        mean, precision = result.x
        sd = 1 / np.sqrt(precision)
        return {"mean": mean, "sd": sd}
    else:
        warnings.warn("Optimization failed.")
        return {"mean": np.nan, "sd": np.nan}

def closed_form_integral(x, mu, sigma):
    """
    Closed-form integral used in computing the systematic error.
    """
    return mu * norm.cdf(x, loc=mu, scale=sigma) - 1 - sigma**2 * norm.pdf(x, loc=mu, scale=sigma)

def closed_form_integral_absolute(mu, sigma):
    """
    Compute the closed-form integral for the absolute systematic error.
    """
    return (closed_form_integral(np.inf, mu, sigma)
            - 2 * closed_form_integral(0, mu, sigma)
            + closed_form_integral(-np.inf, mu, sigma))

def compute_expected_absolute_systematic_error_null(null, alpha=0.05):
    """
    Compute the Expected Absolute Systematic Error (EASE) given the null distribution.
    """
    if null["mean"] == 0 and null["sd"] == 0:
        return 0
    mean = null["mean"]
    sd = null["sd"]
    return closed_form_integral_absolute(mean, sd)

###########################################
# 2. Debiased model and its auxiliary functions
###########################################

def weights_init(m):
    """
    Initialize model weights using Xavier normal initialization.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class M_debiased_multi(nn.Module):
    """
    A multi-layer neural network model with debiasing functionality.
    
    Parameters:
      in_N: Number of input features.
      m: Number of hidden units per layer.
      depth: Number of hidden layers (excluding output layer).
      
    Methods:
      forward: Standard forward propagation without debiasing.
      forward_debias: Forward propagation with debiasing. For each specified layer,
                      the pre-activation output is adjusted by subtracting
                      (alpha * bias_vector).
    """
    def __init__(self, in_N, m, depth=2):
        super(M_debiased_multi, self).__init__()
        self.stack = nn.ModuleList()
        self.stack.append(nn.Linear(in_N, m))
        for i in range(depth - 1):
            self.stack.append(nn.Linear(m, m))
        self.stack.append(nn.Linear(m, 1))
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Standard forward propagation without debiasing
        x = self.act(self.stack[0](x))
        x = self.dropout(x)
        for i in range(1, len(self.stack) - 1):
            x = self.act(self.stack[i](x))
            x = self.dropout(x)
        x = self.stack[-1](x)
        x = 1e-2 + 0.98 * self.sigmoid(x)
        return x

    def forward_debias(self, x, bias_vectors, alphas):
        """
        Forward propagation with debiasing.
        
        Parameters:
          x: Input tensor.
          bias_vectors: Dictionary mapping layer indices to bias vectors.
          alphas: Dictionary mapping layer indices to debiasing coefficients.
          
        For each layer, if the layer index is in bias_vectors, subtract
        (alpha * bias_vector) from the pre-activation output.
        """
        # Layer 0
        a = self.stack[0](x)
        if 0 in bias_vectors:
            a = a - alphas.get(0, 1.0) * bias_vectors[0]
        x = self.act(a)
        x = self.dropout(x)
        # Intermediate layers
        for i in range(1, len(self.stack) - 1):
            a = self.stack[i](x)
            if i in bias_vectors:
                a = a - alphas.get(i, 1.0) * bias_vectors[i]
            x = self.act(a)
            x = self.dropout(x)
        # Output layer (no debiasing)
        x = self.stack[-1](x)
        x = 1e-2 + 0.98 * self.sigmoid(x)
        return x

def compute_bias_vectors(model, X_biased, X_neutral, layer_indices=None):
    """
    Compute the bias vectors for specified layers.
    For each layer i in layer_indices, compute:
      bias_vector_i = mean(pre_activation_i(biased)) - mean(pre_activation_i(neutral))
    
    Parameters:
      model: The trained model (must have a ModuleList 'stack' and activation 'act').
      X_biased: Tensor with biased input samples.
      X_neutral: Tensor with neutral input samples.
      layer_indices: List of layer indices to compute bias vectors for.
                     If None, compute for all layers except the output layer.
    
    Returns:
      A dictionary mapping each selected layer index to its bias vector.
    """
    model.eval()
    bias_vectors = {}
    with torch.no_grad():
        # Layer 0
        a_biased = model.stack[0](X_biased)
        a_neutral = model.stack[0](X_neutral)
        if (layer_indices is None) or (0 in layer_indices):
            bias_vectors[0] = torch.mean(a_biased, dim=0) - torch.mean(a_neutral, dim=0)
        x_biased = model.act(a_biased)
        x_neutral = model.act(a_neutral)
        # Intermediate layers
        for i in range(1, len(model.stack)-1):
            a_biased = model.stack[i](x_biased)
            a_neutral = model.stack[i](x_neutral)
            if (layer_indices is None) or (i in layer_indices):
                bias_vectors[i] = torch.mean(a_biased, dim=0) - torch.mean(a_neutral, dim=0)
            x_biased = model.act(a_biased)
            x_neutral = model.act(a_neutral)
    return bias_vectors

def compute_ease(model, Xt, A, bias_vectors, alphas, W_select, df, column_names_list):
    """
    Compute the Expected Absolute Systematic Error (EASE) for a given debiasing configuration.
    The model uses debiased forward propagation to obtain propensity scores,
    which are then used to calculate inverse probability weights. A weighted
    Poisson regression is performed to estimate logRR and its standard error.
    Finally, the null distribution is fitted and EASE is computed.
    
    Parameters:
      model: The trained debiasing model.
      Xt: Input feature tensor.
      A: Treatment labels (numpy array).
      bias_vectors: Dictionary of bias vectors (keys are layer indices).
      alphas: Dictionary of debiasing coefficients (keys are layer indices).
      W_select: List of column indices used for evaluation.
      df: DataFrame containing the data.
      column_names_list: List of column names in the DataFrame.
      
    Returns:
      The EASE value (lower is better).
    """
    model.eval()
    with torch.no_grad():
        ps = model.forward_debias(Xt, bias_vectors, alphas).squeeze().numpy()
    # Compute inverse probability weights
    wt = np.ones(ps.shape[0])
    wt[A == 1] = 1 / ps[A == 1]
    wt[A == 0] = 1 / (1 - ps[A == 0])
    
    logRR_list = []
    se_list = []
    for i in W_select:
        W = df[column_names_list[i]].to_numpy()
        data = pd.DataFrame({
            'W': W,
            'A': A,
            'weights': wt
        })
        AA = smapi.add_constant(data['A'])
        poisson_model = smapi.GLM(data['W'], AA, 
                                  family=smapi.families.Poisson(),
                                  freq_weights=data['weights']).fit()
        logRR = poisson_model.params['A']
        se_logRR = poisson_model.bse['A']
        logRR_list.append(logRR)
        se_list.append(se_logRR)
    
    logRr = np.array(logRR_list)
    seLogRr = np.array(se_list)
    null_distribution = fit_null(logRr, seLogRr)
    EASE = compute_expected_absolute_systematic_error_null(null_distribution)
    return EASE


class PS(nn.Module):
    def __init__(self, in_N, m, depth=2):
        super(PS, self).__init__()
         
        self.stack = nn.ModuleList() 
        self.stack.append(nn.Linear(in_N, m))
        for i in range(depth-1):
            self.stack.append(nn.Linear(m, m))
        self.stack.append(nn.Linear(m, 1))
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.stack[0](x))
        x = self.dropout(x)   
        for i in range(1,len(self.stack)-1):
            x =  self.act(self.stack[i](x))   
            x = self.dropout(x)   
        x = self.stack[-1](x)
        x = 1e-2+ (0.98)*self.sigmoid(x)       #0.01-0.99
        return  x
    
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def train_NCO(nco, Xt, Wt, weight_1, weight_0):
    criterion = nn.BCELoss()   
    optimizer = optim.Adam(nco.parameters(), lr=0.001, weight_decay=5e-6)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)    
    n_epochs = 25
    batch_size =32
    #
    nco.eval()   
    with torch.no_grad():   
        test_outputs = nco(Xt).squeeze()   
        predicted_prob = test_outputs.numpy()
    actual_treatment = Wt.numpy().squeeze()
    pred = (predicted_prob >=0.5)
#

    for epoch in range(n_epochs):  
        nco.train()  
        running_loss = 0.0
        permutation = torch.randperm(Xt.size(0)) 
        for i in range(0, Xt.shape[0], batch_size):
            ind = permutation[i:i + batch_size]
            batch_x, batch_w = Xt[ind], Wt[ind]
            out = nco(batch_x) # 
        
            optimizer.zero_grad()
            if torch.sum(batch_w):
                out0, out1 = out[batch_w==0], out[batch_w==1]
                bw0, bw1 = batch_w[batch_w==0], batch_w[batch_w==1]
                loss = weight_1*criterion(out[batch_w==1],batch_w[batch_w==1]) + weight_0*criterion(out[batch_w==0],batch_w[batch_w==0])
            else:
                loss = criterion(out,batch_w) 
             
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        scheduler.step()

        nco.eval()   
        with torch.no_grad():   
            test_outputs = nco(Xt).squeeze()   
            predicted_prob = test_outputs.numpy()
        actual_treatment = Wt.numpy().squeeze()
        pred = (predicted_prob >=0.5)
        
    return nco
    
def test_NCO(nco, Xt, Wt):
    nco.eval()  
    with torch.no_grad():
        NCO_pred = nco(Xt).squeeze().numpy()
    
    # Convert probabilities to binary outcomes (0 or 1) based on threshold (0.5)
    NCO_pred_binary = (NCO_pred >= 0.5) 
    act = Wt.numpy().squeeze()

    # Calculate accuracy
    accuracy = np.sum(NCO_pred_binary == act )/act.shape[0]
    print(f'Test Accuracy: {accuracy.item():.4f}')
    return NCO_pred



###########################################
# 3. Example: Grid search for optimal debiasing hyperparameters
###########################################

if __name__ == "__main__":
    # Define input dimension, hidden units, and network depth (number of hidden layers)
    
    
    # load data
    # Create a sample DataFrame and choose columns for evaluation.  load RWD to replace
    file_path = '/project/penncil_im/nco_pruning/df_all_new_outcome_nco_CCSRcat_0305.csv'
    df_all = pd.read_csv(file_path)
    #process data
    rows, cols = df_all.shape

    column_names_list = list(df_all.columns)

    #med_type
    id_values = df_all['med_type'].unique()
    id_counts = df_all['med_type'].value_counts()
    print(id_counts)

    df_all = df_all[df_all['med_type'] != 'DPP4i']
    id_values = df_all['med_type'].unique()
    id_counts = df_all['med_type'].value_counts()
    print(id_counts)

    #treatment
    df_all['treatment'] = df_all['med_type'].map({'GLP-1RA': 1, 'SGLT2i': 0 })
    has_nan = df_all['treatment'].isna().any()
    print(has_nan)
    id_values = df_all['treatment'].unique()
    id_counts = df_all['treatment'].value_counts()
    print(id_counts)

    #age
    id_values = df_all['age_study_start'].unique()
    id_counts = df_all['age_study_start'].value_counts()
    # Group ages into bins:  
    bins = [0, 50, 60, 70, float('inf')]
    labels = ['0-50', '50-60', '60-70','>70']
    df_all['age_group'] = pd.cut(df_all['age_study_start'], bins=bins, labels=labels, right=False)
    # Define function to map age to numerical category
    def map_age_to_numerical(age):
        if age < 50:
            return 0   
        elif 50 <= age < 60:
            return 0.3   
        elif 60 <= age < 70:
            return 0.6
        else:
            return 0.9   
    df_all['age_category_num'] = df_all['age_study_start'].apply(map_age_to_numerical)

    #gender
    id_values = df_all['gender'].unique()
    id_counts = df_all['gender'].value_counts()
    print( id_values)
    print(id_counts)
    # Map 'female' to 1 and 'male' to 0 in the 'gender' column
    df_all['gender'] = df_all['gender'].map({'Female': 1, 'Male': 0})
    has_nan = df_all['gender'].isna().any()
    print(has_nan)


    outcome_prev = []
    j=0
    k = 0
    for i in range(1670,2212+1):  
        prev1 = np.mean(df_all[column_names_list[i]].to_numpy())
        prev2 = np.mean(df_all[df_all[ 'pre_CCSR_'+column_names_list[i][12:] ]==0][column_names_list[i]].to_numpy())
        print(j, i, column_names_list[i], prev1, prev2  )
        j += 1
        if prev2 >=0.005:
            k += 1
            outcome_prev.append(i)
    
    index = 28  #outcome
    outcome_indedx = outcome_prev[index]   
    outcome = column_names_list[outcome_indedx][12:]  
    print('===============', index, outcome, '===============' )
    df = df_all[df_all['pre_CCSR_'+outcome] ==0] 
    Y =  df['CCSR_binary_'+outcome].to_numpy()  
    A =  df['treatment'].to_numpy()  
     #corvariates selection, medical history: 14-420
    column_names_list = list(df.columns)
    pre = []
    pre_high = []
    cov_index = []
    for i in range(14, 420+1):
        cov = df[ column_names_list[i]] 
        pre.append(np.mean(cov) )
        if np.mean(cov)>=0.005:
            cov_index.append(i)
            pre_high.append(np.mean(cov))

    X =  df[[column_names_list[j] for j in  cov_index]].to_numpy() 
    X = np.hstack((X, df[['gender', 'age_category_num']].to_numpy() ))
    Xt = torch.tensor(X, dtype=torch.float32)
    At = torch.tensor(A.reshape(-1, 1), dtype=torch.float32)  
    print(Xt.shape, At.shape)
    
    # Create biased and neutral data for computing bias vectors, replace with NCO split
    # NCO training and split data here
    NCO = []
    j = 0
    for i in range(2395, 2485+1):
        j += 1
        if np.mean(df[column_names_list[i]].to_numpy())>= 0.001:
            NCO.append(i)
    print(len(NCO ))


     ### PS prediction
    d = X.shape[1]
    n_hidden = 300
    n_depth = 3
    model = PS(in_N=d , m= n_hidden, depth= n_depth)   
    model.apply(weights_init)
    print(model) 

    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-6)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)    
    n_epochs = 25
    batch_size =32
    #
    model.eval()   
    with torch.no_grad():   
        test_outputs = model(Xt).squeeze()   
        predicted_prob = test_outputs.numpy()
    actual_treatment = At.numpy().squeeze()
    pred = (predicted_prob >=0.5)
    print('initialzation', np.sum(pred==actual_treatment )/pred.shape[0],optimizer.param_groups[-1]['lr'])
    #
    for epoch in range(n_epochs):  
        model.train()  
        running_loss = 0.0
        permutation = torch.randperm(Xt.size(0)) 
        for i in range(0, Xt.shape[0], batch_size):
            ind = permutation[i:i + batch_size]
            batch_x, batch_a = Xt[ind], At[ind]
            out = model(batch_x)  
            optimizer.zero_grad()
            loss = criterion(out,batch_a)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        model.eval()   
        with torch.no_grad():   
            test_outputs = model(Xt).squeeze()   
            predicted_prob = test_outputs.numpy()
        actual_treatment = At.numpy().squeeze()
        pred = (predicted_prob >=0.5) 
        if (epoch + 1) %1 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(Xt)*batch_size}', np.sum(pred==actual_treatment )/pred.shape[0],optimizer.param_groups[-1]['lr'])

    ### NCO prediction
    NCO = []
    j = 0
    for i in range(2395, 2485+1):
        j += 1
        if np.mean(df[column_names_list[i]].to_numpy())>= 0.001:
            NCO.append(i)
    print(len(NCO ))
    

    #use poisson regression for each W on A:  using propensity score
    model.eval()   
    with torch.no_grad():
        ps = model(torch.tensor(X, dtype=torch.float32)).numpy().squeeze()   
    wt = np.ones(ps.shape[0])
    wt[A==1] = 1/ps[A==1]
    wt[A==0] = 1/(1-ps[A==0])

    column_names_list = list(df.columns)

    logRR_L = []
    se_L = []
    index_all = []

    for i in NCO: 
        W =  df[column_names_list[i]].to_numpy() 
        data = pd.DataFrame({
        'W': W,
        'A': A,
        'weights': wt})
        AA = smapi.add_constant(data['A'])  
        # Fit the Poisson regression model with weights
        poisson_model = smapi.GLM(data['W'], AA, family=smapi.families.Poisson(), freq_weights=data['weights']).fit()
        if poisson_model.bse['A']>2 or poisson_model.params['A']>2.5:
            #print(i,poisson_model.params['A'], poisson_model.bse['A'], '------------')
            continue
        else:
            index_all.append(i) 

        logRR = poisson_model.params['A']
        logRR_L.append( logRR)
        se_logRR = poisson_model.bse['A']
        se_L.append(se_logRR)

    print(len(logRR_L), len(se_L))
    logRr = np.array(logRR_L )  
    seLogRr =np.array(se_L )   
    null_distribution = fit_null(logRr, seLogRr)
    EASE=compute_expected_absolute_systematic_error_null(null_distribution)
    print(EASE)
    
    W_all = [i for i in index_all] 
    print(len(W_all), W_all)
    index_split = random.sample(range(len(W_all)), math.ceil( len(W_all)/3))  
    W_split = [W_all[i] for i in index_split]
    print(len(W_split ), W_split )


    nco_dict = {}
    NCO_select = W_split

    j = 0
    for i in NCO_select:  
        W =  df[column_names_list[i]].to_numpy() 
        print(i, column_names_list[i] )
        Wt =  torch.tensor(W.reshape(-1, 1), dtype=torch.float32)

        weight_1 = 1-np.sum(W)/W.shape[0] 
        weight_0 = 1-weight_1

        n1 =  np.sum(A)
        #treatment 1
        nco1 = PS(in_N=d , m=300, depth=3)   
        nco1.apply(weights_init)
        X1, W1 = X[n1:],W[n1:]
        Xt1, Wt1 = torch.tensor(X1, dtype=torch.float32),torch.tensor(W1.reshape(-1, 1), dtype=torch.float32)
        nco1 = train_NCO(nco1, Xt1, Wt1, weight_1, weight_0)
         
        #treatment 0
        nco0 = PS(in_N=d , m=300, depth=3)   
        nco0.apply(weights_init)
        X0, W0 = X[:n1],W[:n1]
        Xt0, Wt0 = torch.tensor(X0, dtype=torch.float32),torch.tensor(W0.reshape(-1, 1), dtype=torch.float32)
        nco0 = train_NCO(nco0, Xt0, Wt0, weight_1, weight_0)
         
        print('-----------------------'+str(j))
        NCO1_pred = test_NCO(nco1, Xt, Wt)
        NCO0_pred = test_NCO(nco0, Xt, Wt)
        print('-----------------------')
    

        #compare
        nco_dict[str(i)+'nco1_p'] = NCO1_pred
        nco_dict[str(i)+'nco0_p'] = NCO0_pred
        j += 1

    
    #unweighted median
    for i in NCO_select:  
        W =  df[column_names_list[i]].to_numpy()  
        NCO_pred0  =   nco_dict[str(i)+'nco0_p'] 
        NCO_pred1  =  nco_dict[str(i)+'nco1_p'] 
        ind = (abs(NCO_pred0-NCO_pred1)>0.01).reshape(-1,1)  # 
        if i == NCO_select[0]: 
            I = ind
        else:
            I = np.hstack((I,ind))
    print(I.shape)
    avg_I = np.sum(I,axis=1)  
    print(np.max(avg_I))
    con = (avg_I>=(np.max(avg_I)//2   ) ) 
    X_h = Xt[con]
    X_l = Xt[~con]
    A_h = At[con]
    A_l = At[~con]
    print(X_h.shape, X_l.shape,A_h.shape, A_l.shape)
    print('high biased ratio: ', X_h.shape[0]/avg_I.shape[0])



    model_p = M_debiased_multi(in_N=d , m=300, depth=3)
    trained =  model.state_dict()   
    model_p.load_state_dict(trained)

    # Specify the layers for which to compute bias vectors (for example, layers 0 and 1)
    X_biased = X_l
    X_neutral = X_h
    selected_layers = [0, 1]
    bias_vectors = compute_bias_vectors(model_p, X_biased, X_neutral, layer_indices=selected_layers)
    
    # Define the search space for hyperparameters (alphas) for the selected layers.
    # Here we use dictionaries where keys are layer indices.
    alpha_grid = {
        0: np.linspace(0.5, 1.5, 5),
        1: np.linspace(0.5, 1.5, 5)
    }
    
    
    W_left = []
    for i in W_all:
        if i not in W_split:
            W_left.append(i) 
    ### 
    W_select = W_left


    # Grid search for the best alpha values based on minimizing EASE.  other method to fine the best alpha
    min_ease = float('inf')
    best_alphas = {}
    for a0 in alpha_grid[0]:
        for a1 in alpha_grid[1]:
            current_alphas = {0: a0, 1: a1}  # Only for selected layers
            current_ease = compute_ease(model_p, Xt, A, bias_vectors, current_alphas, 
                                        W_select, df, column_names_list)
            if current_ease < min_ease:
                min_ease = current_ease
                best_alphas = current_alphas.copy()
                
    print("Minimum EASE:", min_ease)
    print("Best alphas:", best_alphas)
