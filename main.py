import pandas as pd

test_data = [
    [404, 'Mark', 'Teacher', 1,'a'],
    [404, 'Mark', 'Staff', 1,'b'],
    [404, 'Mark', 'Staff', 1,'a'],
    [659, 'Julio', 'Student', 7,'b'],
    [1025, 'Jasmine', 'Staff', 5,'a'],
    [1025, 'Jasmine', 'Student', 5,'b']
]
cols = ['Unique_ID', 'Name', 'Constinuency Code', 'c','A']

df = pd.DataFrame(test_data, columns=cols)

temp1 = df.groupby(['Unique_ID', 'Name'])['Constinuency Code'].apply(lambda grp: list(grp)).reset_index()
temp2 = df.groupby(['Unique_ID', 'Name'])['c'].apply(lambda grp: list(grp)[0]).reset_index()
temp3 = df.groupby(['Unique_ID', 'Name'])['A'].apply(lambda grp: list(grp)).reset_index()
# temp1 = pd.DataFrame({'text': ['华中科技大学', '武汉大学', '清华大学', '华中科技大学', '武汉大学'],
#                        'label': ["985,理工", "985", "北京", "武汉", "武汉"]})
#
# temp2 = pd.DataFrame({'text': ['华中科技大学', '武汉大学', '清华大学', '华中科技大学', '武汉大学'],
#                        'label': ["985,理工", "985", "北京", "武汉", "武汉"]})
print(temp1)
print(temp2)
print(temp3)
temp = pd.merge(temp1, temp2)
print(temp)
temp = pd.merge(temp,temp3)
print(temp)
