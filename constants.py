# varios ranges for feature generation
subjects = range(15)
trials = range(0, 23)
electrodes = range(62)

# electrode order on mat file
electrode_order_names = [
  'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
  'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
  'C2','C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5',
  'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 
  'O1', 'OZ', 'O2', 'CB2'
]

bands = [
    (1, 4),  # 'delta'
    (4, 8),  # 'theta'
    (8, 12),  # 'alpha'
    (12, 30),  # 'beta'
    (30, 50),  # 'gamma'
]


