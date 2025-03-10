from pandas import read_csv, unique

bandgaps_df = read_csv('https://raw.githubusercontent.com/mannodiarun/perovs_mfml_ga/refs/heads/run_ml/Expt_data.csv')

print(bandgaps_df.head)
print(len(bandgaps_df))
unique_entries = unique(bandgaps_df['Formula'])
print(f'Number of unique entries {unique_entries}')

# Save to
