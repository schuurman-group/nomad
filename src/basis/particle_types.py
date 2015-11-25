import particle

 particle_name =  ['H','D','T','He','Li','Be','B','C','N','O','F','Ne',
                  'Na','Mg','Al','Si','P','S','Cl','Ar']
 particle_width = [4.5, 4.5, 4.5, 0.0,0.0,0.0, 0.0, 22.5,19.5,13.0,8.5,0.0,
                   0.0, 0.0, 0.0, 0.0,0.0,17.5,0.0,0.0]
 particle_mass  = [1.0,2.0,3.0,4.0, 7.0, 9.0,11.0,12.0,14.0,16.0,19.0,20.0,
                   23.0,24.0,27.0,28.0,31.0,32.0,35.45,40.0]
 particle_anum  = [ 1., 1., 1., 2.,  3.,  4., 5., 6., 7., 8., 9., 10.,
                   11., 12., 13., 14., 15., 16., 17., 18]

def valid_particle(particle):
    if(particle.name in atom_types):
        return 1
    else:
	return 0

def load_particle(particle):
    index = particle_name.index(particle.name)
    particle.width = particle_width[index]
    particle.mass  = particle_mass[index]*mass2au
    particle.anum  = particle_anum[index]
    if(particle.width == 0.):
        outfile.write('WARNING: particle '+str(particle.name)+' in library, but width = 0')

def create_particle(pid,dim,name,width,mass):
    new_particle       = particle(dim,pid)
    new_particle.width = width
    new_particle.mass  = mass
    return new_particle

