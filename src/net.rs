use core::str;
use std::{
    fs,
    ops::Add,
    sync::{Arc, RwLock},
};

use ndarray::Array3;
use reqwest::Client;
use tch::{
    nn::{
        self, Conv2D, LinearConfig, Module, ModuleT, Optimizer, OptimizerConfig, VarStore
    },
    Device, Kind, Tensor,
};

use super::train::BATCH_SIZE;

const FEATURES: i64 = 128;

fn init_weights(tensor: &mut Tensor, gain: f64, scale: f64, fan_tensor: Option<&Tensor>) {
    let fan_in = match fan_tensor {
        Some(f) => tch::nn::init::FanInOut::FanIn.for_weight_dims(&f.size()),
        None => tch::nn::init::FanInOut::FanIn.for_weight_dims(&tensor.size()),
    };
    let target_std = scale * gain / (fan_in as f64).sqrt();
    let std = target_std / 0.87962566103423978;
    if std < 1e-10 {
        let _ = tensor.fill_(0.);
    } else {
        // truncated normal distribution
        let l = (1. + statrs::function::erf::erf(-2. / (2_f64).sqrt())) / 2.;
        let u = (1. + statrs::function::erf::erf(2. / (2_f64).sqrt())) / 2.;

        let _ = tensor.uniform_(2. * l - 1., 2. * u - 1.);
        let _ = tensor.erfinv_();
        let _ = tensor.multiply_(&Tensor::from(std * (2_f64).sqrt()));
        let _ = tensor.clamp_(-2. * std, 2. * std);

    }
}

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    nn::conv2d(p, c_in, c_out, ksize, conv2d_cfg)
}

trait QuoriModuleT: ModuleT {
    fn init_weights(&mut self, scale: f64, norm_scale: Option<f64>);
}

#[derive(Debug)]
struct Bias {
    beta: Tensor,
    scale: Option<f64>,
}

impl Bias {
    fn new(p: nn::Path, c_in: i64) -> Bias {
        let beta = p.zeros("bias_beta", &[1, c_in, 1, 1]);
        Bias { beta, scale: None }
    }

    fn set_scale(&mut self, scale: Option<f64>) {
        self.scale = scale;
    }
}

impl ModuleT for Bias {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        match self.scale {
            Some(scale) => xs.add(&self.beta).multiply(&Tensor::from(scale)),
            None => xs.add(&self.beta),
        }
    }
}

#[derive(Debug)]
struct NormFixup {
    beta: Tensor,
    gamma: Option<Tensor>,
    scale: Option<f64>,
}

impl NormFixup {
    fn new(p: nn::Path, c_in: i64, use_gamma: bool) -> NormFixup {
        let beta = p.zeros("norm_beta", &[1, c_in, 1, 1]);
        let gamma = if use_gamma {Some(p.ones("norm_gamma", &[1, c_in, 1, 1]))} else {None};
        NormFixup {
            beta,
            gamma,
            scale: None,
        }
    }

    fn set_scale(&mut self, scale: Option<f64>) {
        self.scale = scale;
    }
}

impl ModuleT for NormFixup {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        match self.scale {
            Some(scale) => {
                match &self.gamma {
                    Some(gamma) => {
                        xs.multiply(&gamma.multiply(&Tensor::from(scale))).add(&self.beta)
                    }
                    None => {
                        xs.multiply(&Tensor::from(scale)).add(&self.beta)
                    }
                }
            }
            None => {
                match &self.gamma {
                    Some(gamma) => {
                        xs.multiply(gamma).add(&self.beta)
                    }
                    None => {
                        xs.add(&self.beta)
                    }
                }
            }
        }
    }
    
}

#[derive(Debug)]
struct GPool {

}

impl Module for GPool {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let layer_mean = xs.mean_dim([2, 3].as_slice(), true, Kind::Float);
        //(layer_max,_argmax) = torch.max(x.view(x.shape[0],x.shape[1],-1).to(torch.float32), dim=2)
        let (mut layer_max, _) = (xs.view((xs.size()[0], xs.size()[1], -1)).to_kind(Kind::Float)).max_dim(2, true);
        layer_max = layer_max.view((xs.size()[0], xs.size()[1], 1, 1));

        let out2 = layer_mean.multiply(&Tensor::from(0.5));
        let out1 = layer_mean;
        let out3 = layer_max;

        Tensor::cat(&[out1, out2, out3], 1)
    }
}

#[derive(Debug)]
struct ValueHeadGPool {

}

impl Module for ValueHeadGPool {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let layer_mean = xs.mean_dim([2, 3].as_slice(), true, Kind::Float);

        let out2 = layer_mean.multiply(&Tensor::from(0.5));
        let out3 = layer_mean.multiply(&Tensor::from(0.24));
        let out1 = layer_mean;

        Tensor::cat(&[out1, out2, out3], 1)
    }
}


#[derive(Debug)]
struct ConvAndGPool {
    conv1r: Conv2D,
    conv1g: Conv2D,
    normg: NormFixup,
    gpool: GPool,
    linearg: nn::Linear,
}

impl ConvAndGPool {
    fn new(p: nn::Path, c_in: i64, c_out: i64, c_gpool: i64) -> ConvAndGPool {
        let conv1r = conv2d(&p / "conv1r", c_in, c_out, 3, 1, 1);
        let conv1g = conv2d(&p / "conv1g", c_in, c_gpool, 3, 1, 1);
        let normg = NormFixup::new(&p / "normg", c_gpool, false);
        let gpool = GPool {};
        let linearg = nn::linear(&p / "linearg", c_gpool * 3, c_out, Default::default());
        ConvAndGPool {
            conv1r,
            conv1g,
            normg,
            gpool,
            linearg,
        }
    }
}

impl ModuleT for ConvAndGPool {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xr = xs.apply(&self.conv1r);
        let xg = xs.apply(&self.conv1g);
        let xg = xg.apply_t(&self.normg, train);
        let xg = xg.relu();
        let xg = xg.apply(&self.gpool).squeeze_dim(-1).squeeze_dim(-1);
        let xg = xg.apply(&self.linearg).unsqueeze(-1).unsqueeze(-1);
        xr.add(&xg)
    }
}

impl QuoriModuleT for ConvAndGPool {
    fn init_weights(&mut self, scale: f64, norm_scale: Option<f64>) {
        let r_scale = 0.8_f64;
        let g_scale = 0.6_f64;
        init_weights(&mut self.conv1r.ws, (2_f64).sqrt(), scale * r_scale, None);
        init_weights(&mut self.conv1g.ws, (2_f64).sqrt(), scale.sqrt() * g_scale.sqrt(), None);
        init_weights(&mut self.linearg.ws, (2_f64).sqrt(), scale.sqrt() * g_scale.sqrt(), None);
    }
    
}

#[derive(Debug)]
struct NormActConv {
    norm: NormFixup,
    conv: Option<Conv2D>,
    convpool: Option<ConvAndGPool>,
}

impl NormActConv {
    fn new(p: nn::Path, c_in: i64, c_out: i64, c_gpool: Option<i64>, ksize: i64, use_gamma: bool) -> NormActConv {
        let norm = NormFixup::new(&p / "norm", c_in, use_gamma);
        let conv;
        let convpool;
        match c_gpool {
            Some(c_gpool) => {
                conv = None;
                convpool = Some(ConvAndGPool::new(&p / "convpool", c_in, c_out, c_gpool));
            }
            None => {
                conv = Some(conv2d(&p / "conv", c_in, c_out, ksize, 1, 1));
                convpool = None;
            }
        }
        NormActConv {
            norm,
            conv,
            convpool,
        }
    }
}

impl ModuleT for NormActConv {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply_t(&self.norm, train);
        let xs = xs.relu();
        return match &self.convpool {
            Some(convpool) => {
                xs.apply_t(convpool, train)
            }
            None => {
                xs.apply(self.conv.as_ref().unwrap())
            }
        }
    }
}

impl QuoriModuleT for NormActConv {
    fn init_weights(&mut self, scale: f64, norm_scale: Option<f64>) {
        self.norm.set_scale(norm_scale);
        match &mut self.convpool {
            Some(convpool) => {
                convpool.init_weights(scale, norm_scale);
            }
            None => {
                init_weights(&mut self.conv.as_mut().unwrap().ws, (2_f64).sqrt(), scale, None);
            }
        }
    }
}

#[derive(Debug)]
struct ResBlock {
    normactconv1: NormActConv,
    normactconv2: NormActConv,
}

impl ResBlock {
    fn new(p: nn::Path, c_main: i64, c_mid: i64, c_gpool: Option<i64>) -> ResBlock {
        let normactconv1 = NormActConv::new(&p / "normactconv1", c_main, c_mid - c_gpool.unwrap_or(0), c_gpool, 3, false);
        let normactconv2 = NormActConv::new(&p / "normactconv2", c_mid - c_gpool.unwrap_or(0), c_main, None, 3, true);
        ResBlock {
            normactconv1,
            normactconv2,
        }
    }
}

impl ModuleT for ResBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let ys = xs.apply_t(&self.normactconv1, train);
        ys.apply_t(&self.normactconv2, train).add(xs)
    }
}

impl QuoriModuleT for ResBlock {
    fn init_weights(&mut self, scale: f64, norm_scale: Option<f64>) {
        self.normactconv1.init_weights(scale, None);
        self.normactconv2.init_weights(0.0, None);
    }
}

#[derive(Debug)]
struct PolicyHead {
    conv1p: Conv2D,
    conv1g: Conv2D,
    biasg: Bias,
    gpool: GPool,
    linear_g: nn::Linear,
    bias2: Bias,
    linear_p: nn::Linear,
}

impl PolicyHead {
    fn new(p: nn::Path, c_in: i64, c_p1: i64, c_g1: i64) -> PolicyHead {
        let conv1p = conv2d(&p / "conv1p", c_in, c_p1, 1, 0, 1);
        let conv1g = conv2d(&p / "conv1g", c_in, c_g1, 1, 0, 1);
        
        let biasg = Bias::new(&p / "biasg", c_g1);
        let gpool = GPool {};

        let linear_g = nn::linear(&p / "linearg", c_g1 * 3, c_p1, Default::default());

        let bias2 = Bias::new(&p / "biasg2", c_p1);
        let linear_p = nn::linear(&p / "linearp", c_p1 * 17 * 17, 132 * 2, Default::default());
        
        PolicyHead {
            conv1p,
            conv1g,
            biasg,
            gpool,
            linear_g,
            bias2,
            linear_p,
        }
    }
}

impl ModuleT for PolicyHead {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xp = xs.apply(&self.conv1p);
        let xg = xs.apply(&self.conv1g);
        let xg = xg.apply_t(&self.biasg, train);
        let xg = xg.relu();
        let xg = xg.apply(&self.gpool).squeeze_dim(-1).squeeze_dim(-1);
        let xg = xg.apply(&self.linear_g).unsqueeze(-1).unsqueeze(-1);
        let x = xp.add(&xg);
        let x = x.apply_t(&self.bias2, train);
        let x = x.relu();
        let x = x.view((-1, 17 * 17 * 32));
        let x = x.apply(&self.linear_p);
        let x1 = x.slice(1, 0, 132, 1);
        let x2 = x.slice(1, 132, 264, 1);
        Tensor::cat(&[x1.log_softmax(1, tch::Kind::Float), x2.log_softmax(1, tch::Kind::Float)], 1)
    }
}

impl QuoriModuleT for PolicyHead {
    fn init_weights(&mut self, scale: f64, norm_scale: Option<f64>) {
        let p_scale = 0.8_f64;
        let g_scale = 0.6_f64;
        let scale_output = 0.3_f64;
        init_weights(&mut self.conv1p.ws, (2_f64).sqrt(), p_scale, None);
        init_weights(&mut self.conv1g.ws, (2_f64).sqrt(), 1.0, None);
        init_weights(&mut self.linear_g.ws, (2_f64).sqrt(), g_scale, None);
        init_weights(&mut self.linear_p.ws, 1.0, scale_output, None);
    }
}

#[derive(Debug)]
struct ValueHead {
    conv1: Conv2D,
    bias1: Bias,
    gpool: GPool,
    linear2: nn::Linear,
    linear_v: nn::Linear,
}

impl ValueHead {
    fn new(p: nn::Path, c_in: i64, c_v1: i64, c_v2: i64) -> ValueHead {
        let conv1 = conv2d(&p / "conv1", c_in, c_v1, 1, 0, 1);
        let bias1 = Bias::new(&p / "bias1", c_v1);
        let gpool = GPool {};
        let mut config = LinearConfig::default();
        config.bias = true;
        let linear2 = nn::linear(&p / "linear2", c_v1 * 3, c_v2, config);
        let linear_v = nn::linear(&p / "linearv", c_v2, 1, config);
        ValueHead {
            conv1,
            bias1,
            gpool,
            linear2,
            linear_v,
        }
    }
}

impl ModuleT for ValueHead {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x = xs.apply(&self.conv1);
        let x = x.apply_t(&self.bias1, train);
        let x = x.relu();
        let x = x.apply(&self.gpool).squeeze_dim(-1).squeeze_dim(-1);
        let x = x.apply(&self.linear2).relu();
        let x = x.apply(&self.linear_v).tanh();
        x
    }
}

impl QuoriModuleT for ValueHead {
    fn init_weights(&mut self, scale: f64, norm_scale: Option<f64>) {
        let bias_scale = 0.2_f64;
        init_weights(&mut self.conv1.ws, (2_f64).sqrt(), 1.0, None);
        init_weights(&mut self.linear2.ws, (2_f64).sqrt(), 1.0, None);
        init_weights(self.linear2.bs.as_mut().unwrap(), (2_f64).sqrt(), bias_scale, Some(&self.linear2.ws));

        init_weights(&mut self.linear_v.ws, 1.0, 1.0, None);
        init_weights(self.linear_v.bs.as_mut().unwrap(), 1.0, bias_scale, Some(&self.linear_v.ws));
    }
}

pub struct ResNetT {
    conv: Conv2D,
    blocks: Vec<ResBlock>,
    norm_trunkfinal: NormFixup,
    policy_head: PolicyHead,
    value_head: ValueHead,
}

unsafe impl Send for ResNetT {}
unsafe impl Sync for ResNetT {}

impl ResNetT {

    pub fn new(p: nn::Path) -> ResNetT {
        let conv = conv2d(&p / "convpre", 14, FEATURES, 3, 1, 1);
        let mut blocks = vec![];
        for i in 0..4 {
            blocks.push(ResBlock::new(&p / format!("block{}", i + 1), FEATURES, FEATURES, None));
        }
        blocks.push(ResBlock::new(&p / "block5", FEATURES, FEATURES, Some(32)));
        for i in 0..2 {
            blocks.push(ResBlock::new(&p / format!("block{}", i + 6), FEATURES, FEATURES, None));
        }
        blocks.push(ResBlock::new(&p / "block8", FEATURES, FEATURES, Some(32)));
        for i in 0..2 {
            blocks.push(ResBlock::new(&p / format!("block{}", i + 9), FEATURES, FEATURES, None));
        }
        let norm_trunkfinal = NormFixup::new(&p / "norm_trunkfinal", FEATURES, false);

        let policy_head = PolicyHead::new(&p / "policy_head", FEATURES, 32, 32);
        let value_head = ValueHead::new(&p / "value_head", FEATURES, 32, 80);
        ResNetT {
            conv,
            blocks,
            norm_trunkfinal,
            policy_head,
            value_head,
        }
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor, Tensor) {
        let mut xs = xs.apply(&self.conv).relu();
        for block in &self.blocks {
            xs = xs.apply_t(block, train);
        }
        let xs = xs.apply_t(&self.norm_trunkfinal, train);
        let xs = xs.relu();
        let p = xs.apply_t(&self.policy_head, train);
        let v = xs.apply_t(&self.value_head, train);
        (p.slice(1, 0, 132, 1), p.slice(1, 132, 264, 1), v)
    }

    pub fn init(&mut self) {
        tch::no_grad(|| {
            let spatial_scale = 0.8;
            init_weights(&mut self.conv.ws, (2_f64).sqrt(), spatial_scale, None);
            let num_blocks = self.blocks.len();
            for blocks in &mut self.blocks {
                let scale = 1. / (num_blocks as f64).sqrt();
                blocks.init_weights(scale, None);
            }
            self.policy_head.init_weights(1.0, None);
            self.value_head.init_weights(1.0, None);
        })
    }
}


pub(crate) struct Net {
    net: ResNetT,
    vs: VarStore,
}

impl Net {
    pub fn new(path: Option<&str>) -> Net {
        let mut vs = VarStore::new(tch::Device::Cuda(0));
        let mut resnet = ResNetT::new(vs.root() / "resnet");
        match path {
            Some(p) => {
                println!("loaded {}", p);
                vs.load(p)
            }
            None => Ok({
                resnet.init()
            }),
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

    pub fn policy_value(
        &self,
        available: Vec<u16>,
        state: Array3<f32>,
        train: bool,
    ) -> (Vec<(i32, f32)>, f32) {
        let mut tensor = tch::Tensor::from_slice(state.as_slice().unwrap())
            .reshape(&[1, 14, 17, 17])
            .to_device(Device::Cuda(0));

        let (mut p_tensor, _, v_tensor) = self.net.forward_t(&tensor, train);

        let (p, v): (&Vec<f32>, f32) = (
            &p_tensor.exp().view([132]).try_into().unwrap(),
            v_tensor.try_into().unwrap(),
        );

        let mut probs = vec![];
        for i in 0..132 {
            if available.contains(&i) {
                probs.push((i as i32, p[i as usize]));
            }
        }

        return (probs, v);
    }

    pub fn save(&self, path: &str) {
        self.vs.save(path).unwrap();
    }
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
        opp_probs: Vec<f32>,
        winner_batch: Vec<f32>,
        lr: f64,
    ) -> (f32, f32, f32) {
        self.optimizer.set_lr(lr);
        self.optimizer.zero_grad();
        let state_tensor = Tensor::from_slice(state_batch.as_slice())
            .reshape(&[BATCH_SIZE.try_into().unwrap(), 14, 17, 17])
            .to_device(Device::Cuda(0));
        let (p, op, v) = self.net.write().unwrap().net.forward_t(&state_tensor, true);
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
        let opp_loss = Net::cross_entropy(
            &op.totype(Kind::Float),
            &Tensor::from_slice(opp_probs.as_slice())
                .reshape(&[BATCH_SIZE.try_into().unwrap(), 132])
                .to_device(Device::Cuda(0)),
        ).multiply(&Tensor::from(0.15));
        let total = value_loss.add(&policy_loss).add(&opp_loss);
        total.backward();
        let vloss: f32 = total.try_into().unwrap();
        let ploss: f32 = policy_loss.try_into().unwrap();
        let oloss: f32 = opp_loss.try_into().unwrap();
        let vloss = vloss - ploss - oloss;
        self.optimizer.step();
        (ploss, oloss, vloss)
    }
    pub fn evaluate_batch(
        &self,
        state_batch: Vec<f32>,
        place_probs: Vec<f32>,
        winner_batch: Vec<f32>,
    ) -> (Tensor, Tensor) {
        let state_tensor = Tensor::from_slice(state_batch.as_slice())
            .reshape(&[BATCH_SIZE.try_into().unwrap(), 14, 17, 17])
            .to_device(Device::Cuda(0));
        let (p, _, v) = self.net.read().unwrap().net.forward_t(&state_tensor, false);
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
