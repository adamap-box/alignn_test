"""Check MP data structure and ordering field"""
import json

print('Loading mp_all_materials.json...')
with open(r'C:\workspace\mp_data\mp_all_materials.json', 'r') as f:
    data = json.load(f)

if isinstance(data, list):
    entries = data
elif isinstance(data, dict):
    entries = data.get('entries', list(data.values()))

print(f'Total entries: {len(entries)}')

# Check first entry structure
print('\nSample entry keys:')
sample = entries[0]
for key in sample.keys():
    val = sample[key]
    val_str = str(val)[:80] if len(str(val)) > 80 else str(val)
    print(f'  {key}: {val_str}')

# Look for magnetic ordering related fields
print('\n\nSearching for magnetic ordering related fields...')
ordering_related = []
for key in sample.keys():
    if any(word in key.lower() for word in ['magnet', 'order', 'spin', 'ferro', 'afm', 'transition']):
        ordering_related.append(key)
        
print(f'Ordering-related fields found: {ordering_related}')

# Check values of ordering fields across entries
if ordering_related:
    print('\nValues of ordering-related fields:')
    from collections import Counter
    for field in ordering_related:
        #values = [e.get(field) for e in entries[:1000]]  # Check first 1000
        values = [e.get(field) for e in entries]
        value_counts = Counter(values)
        print(f'\n{field}:')
        for val, count in value_counts.most_common(10):
            print(f'  {val}: {count}')
