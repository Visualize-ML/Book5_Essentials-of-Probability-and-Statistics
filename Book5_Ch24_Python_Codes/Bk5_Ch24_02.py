

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import seaborn as sns

# Load the iris data
iris_sns = sns.load_dataset("iris") 
# A copy from Seaborn

g = sns.pairplot(iris_sns, kind='reg', diag_kind = 'kde',
                 plot_kws={'line_kws':{'color':'red'}, 
                           'scatter_kws': {'alpha': 0.5}})


g = sns.pairplot(iris_sns, kind='reg', diag_kind = 'kde',
                 hue="species",plot_kws={'scatter_kws': {'alpha': 0.5}})
