''' Ising Model '''
class IsingModel(object):
    def __init__(self, lattice):
        self.lattice = lattice
        self.info = 'Ising({0})'.format(lattice.info)
        self.partitions = lattice.width
        self.region_server = RegionServer(self.partitions)
        self.sample_method = None

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