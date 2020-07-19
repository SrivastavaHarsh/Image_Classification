import global_var as gvr
import global_fe as gfe
import model as mdl
import utly

print('\nStarting program...')
gvr.init()

val = utly.check()
if(val):
    print('\nDataset not found.')
    gfe.createDataset()
else:
    print('\nDataset found.')

mdl.classify()
print('\nProgram executed successfully.')