use std::{
    fs,
    ops::Add,
    sync::{Arc, RwLock},
};

use ndarray::{Array2, Array3, ArrayD};
use rand::Rng;
use reqwest::Client;
use tch::{
    nn::{
        self, BatchNormConfig, Conv, Conv2D, ConvConfig, Func, FuncT, Module, ModuleT, Optimizer,
        OptimizerConfig, SequentialT, VarStore, Variables,
    },
    Device, Kind, Shape, Tensor,
};

use super::train::BATCH_SIZE;

const FEATURES: i64 = 128;

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    nn::conv2d(p, c_in, c_out, ksize, conv2d_cfg)
}

fn downsample(p: nn::Path, c_in: i64, c_out: i64) -> impl ModuleT {
    if 1 != 1 || c_in != c_out {
        nn::seq_t()
            .add(conv2d(&p / "0", c_in, c_out, 1, 0, 1))
            .add(nn::batch_norm2d(&p / "1", c_out, Default::default()))
    } else {
        nn::seq_t()
    }
}

fn batchnorm2d(p: nn::Path, n_features: i64) -> nn::BatchNorm {
    let mut config = BatchNormConfig::default();
    config.momentum = 0.1;
    return nn::batch_norm2d(p, n_features, config);
}

fn res_block(p: nn::Path) -> impl ModuleT {
    let conv1 = conv2d(&p / "conv1", FEATURES, FEATURES, 3, 1, 1);
    let bn1 = batchnorm2d(&p / "bn1", FEATURES);
    let conv2 = conv2d(&p / "conv2", FEATURES, FEATURES, 3, 1, 1);
    let bn2 = batchnorm2d(&p / "bn2", FEATURES);
    nn::func_t(move |xs, train| {
        let ys = xs
            .apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .apply(&conv2)
            .apply_t(&bn2, train);
        (xs + ys).relu()
    })
}

fn basic_layer(p: nn::Path, cnt: i64) -> nn::SequentialT {
    let mut layer = nn::seq_t().add(res_block(&p / "0"));
    for block_index in 1..cnt {
        layer = layer.add(res_block(&p / &block_index.to_string()))
    }
    layer
}

fn policy_head(p: nn::Path) -> impl ModuleT {
    let conv = conv2d(&p / "convp", FEATURES, 16, 1, 0, 1);
    let bn = batchnorm2d(&p / "bnp", 16);
    let linear = nn::linear(&p / "linearp", 17 * 17 * 16, 132, Default::default());
    nn::func_t(move |xs, train| {
        xs.apply(&conv)
            .apply_t(&bn, train)
            .relu()
            .view((-1, 17 * 17 * 16))
            .apply(&linear)
            .log_softmax(1, tch::Kind::Float)
    })
}

fn value_head(p: nn::Path) -> impl ModuleT {
    let conv = conv2d(&p / "convv", FEATURES, 16, 1, 0, 1);
    let bn = batchnorm2d(&p / "bnv", 16);
    let linear = nn::linear(&p / "linearv", 17 * 17 * 16, 512, Default::default());
    let linear2 = nn::linear(&p / "linearv2", 512, 1, Default::default());
    nn::func_t(move |xs, train| {
        xs.apply(&conv)
            .apply_t(&bn, train)
            .relu()
            .view((-1, 17 * 17 * 16))
            .apply(&linear)
            .relu()
            .apply(&linear2)
            .tanh()
    })
}

pub struct ResNetT<'a> {
    f: Box<dyn 'a + Fn(&Tensor, bool) -> (Tensor, Tensor) + Send>,
}

unsafe impl Send for ResNetT<'_> {}
unsafe impl Sync for ResNetT<'_> {}

impl<'a> std::fmt::Debug for ResNetT<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ResNetT")
    }
}

pub fn func_t<'a, F>(f: F) -> ResNetT<'a>
where
    F: 'a + Fn(&Tensor, bool) -> (Tensor, Tensor) + Send,
{
    ResNetT { f: Box::new(f) }
}

impl<'a> ResNetT<'a> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        (*self.f)(xs, train)
    }
}

fn resnet1(p: &nn::Path) -> ResNetT<'static> {
    let p = p;
    let conv = conv2d(p / "convpre", 9, FEATURES, 1, 0, 1);
    let bn = batchnorm2d(p / "bnmain", FEATURES);
    let blocks = basic_layer(p / "layer", 10);
    let p_head = policy_head(p / "phead");
    let v_head = value_head(p / "vhead");
    func_t(move |xs, train| {
        let ys = xs.apply(&conv).apply_t(&bn, train).apply_t(&blocks, train);
        return (ys.apply_t(&p_head, train), ys.apply_t(&v_head, train));
    })
}

fn resnet(p: &nn::Path) -> FuncT<'static> {
    let conv = conv2d(p / "convpre", 4, FEATURES, 1, 0, 1);
    let blocks = basic_layer(p / "layer", 4);
    let p_head = policy_head(p / "phead");
    let v_head = value_head(p / "vhead");
    nn::func_t(move |xs, train| {
        let ys = xs.apply(&conv).apply_t(&blocks, train);
        Tensor::concat(&[ys.apply_t(&p_head, train), ys.apply_t(&v_head, train)], 1)
    })
}

pub(crate) struct Net {
    net: ResNetT<'static>,
    vs: VarStore,
}

impl Net {
    pub fn new(path: Option<&str>) -> Net {
        let mut vs = VarStore::new(tch::Device::Cuda(0));
        let resnet = resnet1(&vs.root());
        match path {
            Some(p) => {
                println!("loaded {}", p);
                vs.load(p)
            }
            None => Ok(()),
        }
        .expect("LOAD FAILED");
        Net { net: resnet, vs }
    }
    pub fn cross_entropy(input: &Tensor, target: &Tensor) -> Tensor {
        let size = input.size()[0];
        // println!("size: {:?}", input.size());
        let input = input.view([size * 132]);
        // let v: Vec<f32> = input.view([size*81]).into();
        // println!("{:?}", v);
        let target = target.view([size * 132]);
        -input.dot(&target) / size
    }

    pub fn policy_value_loss(&self, state: Vec<f32>, prob: Vec<f32>, win: f32) -> (f32, f32) {
        let tensor = tch::Tensor::from_slice(state.as_slice())
            .to_device(Device::Cuda(0))
            .reshape(&[1, 9, 17, 17]);
        let (p_tensor, v_tensor) = self.net.forward_t(&tensor, false);
        let v_loss = v_tensor.mse_loss(
            &Tensor::from(win).to_device(Device::Cuda(0)),
            tch::Reduction::Mean,
        );
        let p_loss = Net::cross_entropy(
            &p_tensor.to_kind(Kind::Float).reshape(&[1, 132]),
            &Tensor::from_slice(prob.as_slice())
                .to_device(Device::Cuda(0))
                .reshape(&[1, 132]),
        );
        (p_loss.try_into().unwrap(), v_loss.try_into().unwrap())
        /*let rot = rand::thread_rng().gen_range(0, 4);
        tensor = tensor.rot90(rot, &[2, 3]);
        let flip = rand::thread_rng().gen_bool(0.5);
        if flip {
            tensor = tensor.flip(&[2]);
        }
        let (mut p_tensor, v_tensor) = self.net.forward_t(&tensor, false);
        p_tensor = p_tensor.reshape(&[9, 9]);
        if flip {
            p_tensor = p_tensor.flipud();
        }
        p_tensor = p_tensor.rot90(-rot, &[0, 1]);
        (p_tensor, v_tensor)*/
        /*
        //p_tensor = p_tensor.reshape(&[81]);
        let (p, v): (&Vec<Vec<f32>>, &Vec<f32>) = (&p_tensor.exp().into(), v_tensor.into());
        //Array2::from(p);
        let mut probs = vec![];
        for i in 0..81 {
            if available.contains(&i) {
                probs.push(p[i as usize]);
            } else {
                probs.push(0.);
            }
        }
        let pr = probs
            .iter()
            .enumerate()
            .map(|(pos, prb)| (pos as i32, *prb))
            .collect();
        return (pr, v);
         */
    }
    pub fn policy_value(
        &self,
        available: Vec<u16>,
        state: Array3<f32>,
        train: bool,
    ) -> (Vec<(i32, f32)>, f32) {
        let mut tensor = tch::Tensor::from_slice(state.as_slice().unwrap())
            .reshape(&[1, 9, 17, 17])
            .to_device(Device::Cuda(0));
        //let flip = rand::thread_rng().gen_bool(0.5);
        //if flip {
        //    tensor = tensor.flip(&[2]);
        //}
        //let vc: Vec<f32> = tensor.abs().into();
        //let ar3 = Array3::from_shape_vec((7, 9, 9), vc);
        //println!("{:?}", ar3);
        let (mut p_tensor, v_tensor) = self.net.forward_t(&tensor, train);
        //p_tensor = rot90_action(p_tensor, 0, flip);
        //p_tensor = p_tensor.reshape(&[81]);
        let (p, v): (&Vec<f32>, f32) = (
            &p_tensor.exp().view([132]).try_into().unwrap(),
            v_tensor.try_into().unwrap(),
        );
        //Array2::from(p);
        let mut probs = vec![];
        for i in 0..132 {
            if available.contains(&i) {
                probs.push((i as i32, p[i as usize]));
            }
        }
        //println!("{:?}", probs);
        return (probs, v);
    }

    pub fn save(&self, path: &str) {
        self.vs.save(path).unwrap();
    }
}

fn rot90_action(tensor: Tensor, rot: i64, flip: bool) -> Tensor {
    // tensor: 3*8 = 24
    // rot: 0, 1, 2, 3
    // flip: true, false
    let mut tensor = tensor.reshape(&[3, 8]);
    // in each row, 0 is up, 1 is up-right, 2 is right, etc.
    // so flip is exchange 0 and 4, 1 and 3, 7 and 5
    if flip {
        tensor = tensor.index_select(
            1,
            &Tensor::from_slice(&[4, 3, 2, 1, 0, 7, 6, 5])
                .to_kind(Kind::Int64)
                .to_device(Device::Cuda(0)),
        );
    }
    // rot90 is actually shift 2 positions to the right
    tensor = tensor.roll(&[2 * rot], &[1]);
    tensor.reshape(&[24])
}

pub(crate) struct NetTrain {
    pub net: Arc<RwLock<Net>>,
    optimizer: Optimizer,
}

impl NetTrain {
    pub fn new(path: Option<&str>) -> NetTrain {
        let net = Net::new(path);
        let optimizer = tch::nn::sgd(0.9, 0., 1e-4, false)
            .build(&net.vs, 2e-3)
            .unwrap();
        NetTrain {
            net: Arc::new(net.into()),
            optimizer,
        }
    }
    pub fn train_step(
        &mut self,
        state_batch: Vec<f32>,
        place_probs: Vec<f32>,
        winner_batch: Vec<f32>,
        lr: f64,
    ) -> (f32, f32) {
        self.optimizer.set_lr(lr);
        self.optimizer.zero_grad();
        let state_tensor = Tensor::from_slice(state_batch.as_slice())
            .reshape(&[BATCH_SIZE.try_into().unwrap(), 9, 17, 17])
            .to_device(Device::Cuda(0));
        let (p, v) = self.net.write().unwrap().net.forward_t(&state_tensor, true);
        let value_loss = v.mse_loss(
            &Tensor::from_slice(winner_batch.as_slice())
                .reshape(&[BATCH_SIZE.try_into().unwrap(), 1])
                .to_device(Device::Cuda(0)),
            tch::Reduction::Mean,
        );
        //.multiply(&Tensor::from(3.));
        let policy_loss = Net::cross_entropy(
            &p.totype(Kind::Float),
            &Tensor::from_slice(place_probs.as_slice())
                .reshape(&[BATCH_SIZE.try_into().unwrap(), 132])
                .to_device(Device::Cuda(0)),
        );
        let total = value_loss.add(&policy_loss);
        total.backward();
        let vloss: f32 = total.try_into().unwrap();
        let ploss: f32 = policy_loss.try_into().unwrap();
        let vloss = vloss - ploss;
        self.optimizer.step();
        (ploss, vloss)
    }
    pub fn evaluate_batch(
        &self,
        state_batch: Vec<f32>,
        place_probs: Vec<f32>,
        winner_batch: Vec<f32>,
    ) -> (Tensor, Tensor) {
        let state_tensor = Tensor::from_slice(state_batch.as_slice())
            .reshape(&[BATCH_SIZE.try_into().unwrap(), 9, 17, 17])
            .to_device(Device::Cuda(0));
        let (p, v) = self.net.read().unwrap().net.forward_t(&state_tensor, false);
        return (p.exp(), v);
    }
    pub async fn save(&self, path: &str, http_address: String) {
        self.net.write().unwrap().vs.save(path).unwrap();
        if http_address.is_empty() {
            return;
        }
        let file = fs::read(path).unwrap();
        let client = Client::new();
        let res = client.post(http_address).body(file).send().await.unwrap();

        let status = res.status();
        if status.is_success() {
            println!("Model uploaded successfully");
        } else {
            println!("Model save failed");
        }
    }
}
