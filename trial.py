''' Physical System '''
class FreeFermion(object):
# input:
#     mass :: float : mass of the fermions, in [-1.,1.]
#     size :: int : length of the 1D many-body state
    def __init__(self, size, mass, c = 1.):
        assert -1. <= mass <= 1., 'mass must be in the range of [-1.,1.].'
        assert size%2 == 0, 'size must be even.'
        self.size = size
        self.mass = mass # fermion mass
        self.c = c # central charge
        self.info = 'FF({0},{1:.2f},{2:.1f})'.format(self.size, self.mass, self.c)
        self._built = False
        self.entropy = None

    # build system
    def build(self):
        if not self._built:
            self._built = True # change status
            # construct single-particle density matrix
            u = np.tile([1.+self.mass, 1.-self.mass], self.size//2) # hopping
            A = np.roll(np.diag(u), 1, axis=1) # periodic rolling
            A[-1,0] = -A[-1,0] # antiperiodic boundary condition, to avoid zero mode
            A = 1j * (A - np.transpose(A)) # Hamiltonian
            (_, U) = np.linalg.eigh(A) # digaonalization
            V = U[:,:self.size//2] # take lower half levels
            self.G = np.dot(V, V.conj().T) # construct density matrix

    # calculate entanglement entropy given sites
    def renyi(self, sites):
        if len(sites) == 0: return 0.
        # diagonalize reduced density matrix
        p = np.linalg.eigvalsh(self.G[sites,:][:,sites])
        # return 2nd Renyi entropy
        renyi = -self.c * np.sum(np.log(np.abs(p**2 + (1.- p)**2))).item()
        return renyi


''' Ising Model '''
class IsingModel(object):
    def __init__(self, lattice):
        self.lattice = lattice
        self.info = 'Ising({0})'.format(lattice.info)
        # self.partitions = lattice.width
        # self.region_server = RegionServer(self.partitions)
        # self.sample_method = None
        
    # get Ising configuration using the specified sampling method
    def ising_config(self, sample_method = 'weighted', batch = 1):
        if sample_method != self.sample_method: # if method changed
            self.sample_method = sample_method # update method state
            self.region_server.fill(sample_method) # fill the server by new method
        regions = self.region_server.fetch(batch)
        # prepare empty lists for configs
        confs = []
        # go through all regions in the batch
        for region in regions:
            # configuration of Ising boundary
            confs.append(region.config())

        return tf.convert_to_tensor(np.array(confs), dtype=tf.float64, name = 'confs')

    # build model (given input ising configurations and log bound dimension lnD )
    def build(self, confs, lnD):
        self.lattice.build() # first build lattice
        # setup adjacency tensors as TF constants
        A_bg = tf.constant(self.lattice.adjten('1'),dtype = tf.float64, name = 'A_bg')
        As_h = tf.constant(self.lattice.adjten('wh'),dtype = tf.float64, name = 'As_h')
        As_J = tf.constant(self.lattice.adjten('wJ'),dtype = tf.float64, name = 'As_J')
        # boundary Ising configurations
        conf0 = tf.ones([self.lattice.size['wh']], dtype = tf.float64, name = 'conf0')

        # external field configurations
        with tf.name_scope('h'):
            self.h = tf.dtypes.cast(lnD/2., tf.float64) # external field strength
            hs = self.h * confs
            h0 = self.h * conf0

        # coupling strength (trainable variable)
        self.J = tf.Variable(0.27 * np.ones(self.lattice.size['wJ']), 
                dtype = tf.float64, name = 'J')

        # generate weighted adjacency matrices
        with tf.name_scope('Ising_net'):
            A_J = self.wdotA(self.J, As_J, name = 'A_J')
            A_hs = self.wdotA(hs, As_h, name = 'A_hs')
            A_h0 = self.wdotA(h0, As_h, name = 'A_h0')
            # construct full adjacency matrix
            As = A_hs + A_bg + A_J
            A0 = A_h0 + A_bg + A_J

        # calcualte free energy and model entropy
        with tf.name_scope('free_energy'):
            self.Fs = self.free_energy(As, self.J, hs, name = 'Fs')
            self.F0 = self.free_energy(A0, self.J, h0, name = 'F0')

        # Smdl denotes the entropy of Ising model
        self.Smdl = tf.subtract(self.Fs, self.F0, name = 'S_mdl')
        #print(self.Smdl.numpy(), self.Fs.numpy(), self.F0.numpy())
        
        # calculate cost function
        # Ssys is the entropy of physical system
        # with tf.name_scope('cost'):
        #     self.MSE = tf.reduce_mean(input_tensor=tf.square(self.Smdl/self.Ssys - 1.))
        #     self.wall = tf.reduce_sum(input_tensor=tf.nn.relu(self.J[1:]-self.J[:-1]))
        #     self.cost = self.MSE
        #     # record cost function
        #     tf.summary.scalar('logMSE', tf.math.log(self.MSE))
        #     tf.summary.scalar('logwall', tf.math.log(self.wall + 1.e-10))

        # coupling regularizer
        with tf.name_scope('regularizer'):
            Jrelu = tf.nn.relu(self.J) # first remove negative couplings
            # construct the upper bond
            Jmax = tf.concat([tf.reshape(self.h,[1]), Jrelu[:-1]], axis=0)
            # clip by the upper bond and assign to J
            self.regularizer = self.J.assign(tf.minimum(Jrelu, Jmax))

    # convert coupling to weight, and contract with adjacency matrix
    def wdotA(self, f, A, name = 'wdotA'):
        with tf.name_scope(name):
            return tf.tensordot(tf.exp(2. * f), A, axes = 1)

    # free energy (use logdet)
    def free_energy(self, A, J, h, name = 'F'):
        with tf.name_scope(name):
            with tf.name_scope('Jh'):
                Js = J * tf.constant(self.lattice.lcvect('wJ'), name = 'J_count')
                hs = h * tf.constant(self.lattice.lcvect('wh'), name = 'h_count')
                F0 = tf.reduce_sum(input_tensor=Js) + tf.reduce_sum(input_tensor=hs, axis=-1)
            logdetA = tf.linalg.slogdet(A)[1]
            F = -0.5*logdetA + F0

        return F   

''' Entanglement Feature Learning
input:
    system : object that has a method renyi to return the renyi entropy
    model : Ising model where the coupling strengths J over different layers as trainable parameters
'''
# entanglement feature data server
class DataServer(object):
    def __init__(self, model, system, sample_method='weighted'):
        self.model = model
        self.system = system
        self.sample_method = sample_method
        self.size = system.size
        self.partitions = model.lattice.width
        assert self.size%self.partitions == 0, 'Size not divisible by partitions.'
        self.blocksize = self.size // self.partitions
        self.lnD = self.blocksize * self.system.c * np.log(2.). # bound dimension of RTN state
        # set up a entanglement region server
        self.region_server = RegionServer(self.partitions)        
        self.blockmap = None
        self.ising_config = None  # Ising configurations to be sampled
        self.model_entropy = None # corresponding Renyi entropy of physical model

    def __getattr__(self, attr):
        if attr == 'info':
            return self.model.info+self.system.info+''.join(str(x) for x in self.sample_method)
        else:
            raise AttributeError("%s object has no attribute named %r" %
                         (self.__class__.__name__, attr))

    # get sites given a region
    def sites(self, region):
        # if the region is over half of the partitions
        if len(region) > self.partitions//2:
            # take the complement region instead
            return self.sites(region.complement())
        else: # map block indices to site indices
            if self.blockmap is None:
                self.blockmap = np.arange(self.size).reshape(
                            [self.partitions, self.blocksize])
            return self.blockmap[list(region)].flatten()

    # get Ising configuration using the specified sampling method
    def configurations(self, sample_method = 'weighted', batch = 1):
        if sample_method != self.sample_method: # if method changed
            self.sample_method = sample_method # update method state
        self.region_server.fill(sample_method) # fill the server by new method
        regions = self.region_server.fetch(batch)
        # prepare empty lists for configs
        confs = []
        Renyi = []
        # go through all regions in the batch
        for region in regions:
            # configuration of Ising boundary
            confs.append(region.config())
            # entanglement entropy from system
            Renyi.append(self.system.renyi(self.sites(region)))
        self.ising_config = tf.convert_to_tensor(np.array(confs), dtype=tf.float64, name = 'confs')
        #self.ising_config = np.array(confs)
        self.model_entropy = np.array(Renyi)
        # return tf.convert_to_tensor(np.array(confs), dtype=tf.float64, name = 'confs')
        #return {'ising_config': np.array(confs), 'model_entropy': np.array(Renyi)}

    # # calculate entanglement feature and package data
    # def configurations(self, regions):
    #     # prepare empty lists for configs and 2nd Renyi entropy
    #     confs = []
    #     Renyi = []
    #     # go through all regions in the batch
    #     for region in regions:
    #         # configuration of Ising boundary
    #         confs.append(region.config())
    #         # entanglement entropy from system
    #         Renyi.append(self.system.renyi(self.sites(region)))
    #     # return data as a dict
    #     return np.array([np.array(confs), np.array(Renyi)])



    # fetch data
    def fetch(self, method, batch = None):
        if method != self.method: # if method changed
            self.method = method # update method state
        self.region_server.fill(method) # fill the server by new method
        
        return self.pack(self.region_server.fetch(batch))


# EFL machine
from datetime import datetime
class Machine(object):
    def __init__(self, model, system, sample_method='weighted'):
        self.model = model
        self.system = system
        self.sample_method = sample_method
        self.data_server = DataServer(model, system)
        self.graph = tf.Graph() # TF graph
        # self.session = tf.compat.v1.Session(graph = self.graph) # TF session
        self.para = None # parameter dict
        # status flags
        self._built = False
        self.cost = None
        self.learning_rate = None
        self.beta1 = None
        self.beta2 = None
        self.epsilon = None


    def __getattr__(self, attr):
        if attr == 'info':
            return self.model.info+self.system.info+''.join(str(x) for x in self.sample_method)
        else:
            raise AttributeError("%s object has no attribute named %r" %
                         (self.__class__.__name__, attr))

    def cost_function(self):
        # cost function to be defined here
        pass

    # build machine
    def build(self):
        if not self._built: # if not built
            self._built = True # change status
            # build machine
            self.system.build() # build system
            # add nodes to the TF graph
            with self.graph.as_default():
                self.model.build(self.data_server.lnD) # build model
                self.step = tf.Variable(0, name='step', trainable=False)
                self.optimizer = tf.optimizers.Adam(
                                learning_rate=self.learning_rate, 
                                beta1=self.beta1, 
                                beta2=self.beta2, 
                                epsilon=self.epsilon)
                self.trainer = self.optimizer.minimize(self.cost, 
                                global_step = self.step)
                self.regularizer = self.model.regularizer
                self.writer = self.pipe() # set up data pipeline
                self.saver = tf.compat.v1.train.Saver() # add saver

    # pipe data (by summary)
    def pipe(self):
        # get variable names
        var_names = {i:name for name, i in self.model.lattice.slot_dict['wJ'].items()}
        # go through each component of J
        for i in range(self.model.lattice.size['wJ']):
            tf.compat.v1.summary.scalar('J/{0}'.format(var_names[i]), self.model.J[i])
        # optimizer slots
        slot_names = self.optimizer.get_slot_names()
        for slot_name in slot_names:
            slot = self.optimizer.get_slot(self.model.J, slot_name)
            for i in range(self.model.lattice.size['wJ']):
                tf.compat.v1.summary.scalar('{0}/{1}'.format(slot_name, var_names[i]), slot[i])
        self.summary = [tf.compat.v1.summary.merge_all(), self.step]
        timestamp = datetime.now().strftime('%d%H%M%S')
        return tf.compat.v1.summary.FileWriter('./log/' + timestamp)

           
    # train machine
    def train(self, steps=1, check=20, sample_method=None, batch=None, 
            learning_rate=0.005, beta1=0.9, beta2=0.9, epsilon=1e-8):
        self.build() # if not built, build it
        if sample_method is None:
            sample_method = self.sample_method # by default, use global sample method
        else:
            self.sample_method = sample_method # otherwise sample method updated
        # setup parameter feed dict
        self.para = {self.learning_rate:learning_rate, 
                    self.beta1:beta1, 
                    self.beta2:beta2, 
                    self.epsilon:epsilon}

        # start training loop
        for i in range(steps):
            # construct the feed dict, attach data to para
            self.feed = {**self.para, **self.data_server.fetch(sample_method, batch)}
            try: # zero determinant may cause a problem, try it
                self.session.run(self.trainer, self.feed) # train one step
            except tf.errors.InvalidArgumentError: # when things go wrong
                continue # skip the rest, go the the next batch of data
            self.session.run(self.regularizer) # run regularization
            if self.session.run(self.step)%check == 0: # summarize
                self.writer.add_summary(*self.session.run(self.summary, self.feed))

    # graph export for visualization in TensorBoard 
    def add_graph(self):
        self.writer.add_graph(self.graph) # writter add graph

    # save session
    def save(self):
        # save model, without saving the graph
        path = self.saver.save(self.session, './machine/'+self.info, 
                                write_meta_graph=False)
        print('INFO:tensorflow:Saving parameters to %s'%path)

    # load session
    def load(self):
        self.build() # if not built, build it
        # restore model
        self.saver.restore(self.session, './machine/'+self.info)
        # session is initialized after loading 
        self._initialized = True