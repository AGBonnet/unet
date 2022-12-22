import tensorflow as tf
import os

def report_resources():
   print()
   print(Colors.RED + Colors.BOLD + Colors.UNDERLINE + 'Resources' + Colors.END)
   print('Current path:', os.path.abspath(os.getcwd()))
   print('GPU available: ', tf.config.list_physical_devices('GPU'))
   print('TensorFlow Version: ', tf. __version__)
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
   #device = torch.device('cpu')
   #if torch.cuda.is_available(): device = torch.device('cuda')
   #if torch.has_mps: device = torch.device('mps')

class Colors:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
