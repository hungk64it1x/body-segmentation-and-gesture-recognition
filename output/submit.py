import pandas as pd

sub1 = pd.read_csv('../segmentation/csv/sub1.csv')
sub2 = pd.read_csv('../cls/csv/sub2.csv')

columns = sub1.columns.tolist()
sub = pd.DataFrame(columns=columns)
sub = pd.concat([sub1, sub2])
sub.to_csv('results.csv', index=False)