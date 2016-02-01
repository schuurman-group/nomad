import numpy as np
import src.fmsio.glbl as glbl
import src.fmsio.fileio as fileio
import src.basis.trajectory as trajectory
import src.spawn.utilities as utilities
# This is the top-level routine for spawning
#
# Schematic
#
#   start, ti
#     |           if Min(parent.overlap(traj(st',ti))) < Omin
# parent(r,p,st,ti) ------------------------------------> child(r,p',st',ti)
#                                 
# 1. if the minimum overlap between the "parent" trajectory on state st with trajectories
#    on state st' drops below a user-defined threshold, spawn a new function on st' at the
#    same position, with a scaled momentum to enforce constraint that classical energies
#    be equal. 
#
def spawn(master,current_time,dt):

    #
    #
    #
    for i in range(master.n_total()):
      
        if not master.traj[i].alive:
            continue

        parent = master.traj[i]
  
        for st in range(master.nstates):
      
            # don't check overlap with basis functions on same state
            if st == parent.state:
                continue

            s_array = [abs(parent.overlap(master.traj[j],st_orthog=False)) 
                         if master.traj[j].state==st and master.traj[j].alive 
                         else 0. 
                         for j in range(master.n_total())]

            print('s_array='+str(s_array))
            if min(s_array) < glbl.fms['continuous_min_overlap']:
                child        = trajectory.copy_traj(parent)
                child.state  = st
                child.parent = parent.tid

                print("scale_dir="+str(parent.derivative(st)))
                success = utilities.adjust_child(parent, child, parent.derivative(st))
                sij = parent.overlap(child) 
 
                # try to set up the child
                if not success:
                    pass
                elif abs(sij) < glbl.fms['spawn_olap_thresh']:
                    fileio.print_fms_logfile('spawn_bad_step',
                                            ['child-parent overlap too small'])
                else:
                    child_created = True
                    spawn_time = current_time
                    parent.last_spawn[child.state] = spawn_time
                    child.last_spawn[parent.state] = spawn_time

                    bundle_overlap = utilities.overlap_with_bundle(child,master)
                    if not bundle_overlap:
                        master.add_trajectory(child)
                        fileio.print_fms_logfile('spawn_success',[current_time,parent.tid,st])
                        utilities.write_spawn_log(current_time, current_time, current_time, parent, master.traj[-1])
                    else:
                        fileio.print_fms_logfile('spawn_bad_step',['overlap with bundle too large'])

