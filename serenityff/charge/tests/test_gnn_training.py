import os
from typing import OrderedDict

import pytest
from numpy import array_equal
from rdkit import Chem
from torch import device, load
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.cuda import is_available


from serenityff.charge.gnn.training import Trainer
from serenityff.charge.gnn.utils import (
    ChargeCorrectedNodeWiseAttentiveFP,
    CustomData,
    get_graph_from_mol,
)
from serenityff.charge.utils import NotInitializedError


@pytest.fixture
def cwd() -> str:
    return os.path.dirname(__file__)


@pytest.fixture
def sdf_path(cwd) -> str:
    return f"{cwd}/../data/example.sdf"


@pytest.fixture
def pt_path(cwd) -> str:
    return f"{cwd}/../data/example_graphs.pt"


@pytest.fixture
def model_path(cwd) -> str:
    return f"{cwd}/../data/example_model.pt"


@pytest.fixture
def statedict_path(cwd) -> str:
    return f"{cwd}/../data/example_state_dict.pt"


@pytest.fixture
def statedict(statedict_path) -> OrderedDict:
    return load(statedict_path, map_location="cpu")


@pytest.fixture
def molecule(sdf_path) -> CustomData:
    return Chem.SDMolSupplier(sdf_path, removeHs=False)[0]


@pytest.fixture
def graph(molecule) -> CustomData:
    return get_graph_from_mol(molecule, index=0, sdf_property_name="MBIScharge")


@pytest.fixture
def model() -> ChargeCorrectedNodeWiseAttentiveFP:
    return ChargeCorrectedNodeWiseAttentiveFP()


@pytest.fixture
def optimizer(model):
    return Adam(model.parameters(), lr=0.001)


@pytest.fixture
def lossfunction():
    return mse_loss


@pytest.fixture
def trainer(model, optimizer):
    trainer = Trainer()
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.save_prefix = os.path.dirname(__file__) + "/test"
    return trainer


def test_init_and_forward_model(model, graph) -> None:
    model = model
    model.train()
    out = model(
        graph.x,
        graph.edge_index,
        graph.batch,
        graph.edge_attr,
        graph.molecule_charge,
    )
    assert len(out) == 18
    return


def test_initialize_trainer(trainer, model, sdf_path, pt_path, statedict_path, model_path, statedict) -> None:
    # test init
    assert trainer.device == device("cuda") if is_available() else device("cpu")
    trainer.device = "CPU"
    trainer.device = "cpu"
    trainer.device = device("cpu")

    # test setters
    trainer.model = statedict_path
    trainer.model = model_path
    trainer.model = statedict
    trainer.model = model
    with pytest.raises(FileNotFoundError):
        trainer.model = "faulty"
    with pytest.raises(TypeError):
        trainer.model = 3213
    with pytest.raises(TypeError):
        trainer.optimizer = "faulty"
    with pytest.raises(TypeError):
        trainer.loss_function = "faulty"
    with pytest.raises(ValueError):
        trainer.device = "faulty value"
    with pytest.raises(TypeError):
        trainer.device = 2

    trainer.save_prefix = os.path.dirname(__file__)
    trainer.save_prefix = os.path.dirname(__file__) + "/test/testprefix"
    trainer.save_model_statedict()
    assert os.path.isfile(os.path.dirname(__file__) + "/test/testprefix_model_sd.pt")
    assert os.path.isdir(os.path.dirname(__file__) + "/test")
    os.remove(os.path.dirname(__file__) + "/test/testprefix_model_sd.pt")
    os.rmdir(os.path.dirname(__file__) + "/test")
    trainer.save_prefix = os.path.dirname(__file__)
    trainer.save_prefix = os.path.dirname(__file__) + "/test/testprefix"
    trainer.save_model()
    assert os.path.isfile(os.path.dirname(__file__) + "/test/testprefix_model.pt")
    assert os.path.isdir(os.path.dirname(__file__) + "/test")
    os.remove(os.path.dirname(__file__) + "/test/testprefix_model.pt")
    os.rmdir(os.path.dirname(__file__) + "/test")

    # test graph creation
    trainer.gen_graphs_from_sdf(sdf_path)
    assert len(trainer.data) == 20
    trainer.load_graphs_from_pt(pt_path)
    assert len(trainer.data) == 3
    return


def test_prepare_train_data(trainer, sdf_path):
    with pytest.warns(Warning):
        trainer.prepare_training_data()
    trainer.gen_graphs_from_sdf(sdf_path)
    trainer.prepare_training_data()
    trainer.prepare_training_data(split_type="kfold", n_splits=3)
    with pytest.raises(NotImplementedError):
        trainer.prepare_training_data(split_type="faulty")
    return


def test_train_model(trainer, sdf_path) -> None:
    trainer.gen_graphs_from_sdf(sdf_path)
    trainer.prepare_training_data(train_ratio=0.5)
    trainer.train_model(epochs=1)
    for file in [
        f"{trainer.save_prefix}_train_loss.npy",
        f"{trainer.save_prefix}_eval_loss.npy",
        f"{trainer.save_prefix}_model_sd.pt",
    ]:
        assert os.path.isfile(file)
        os.remove(file)
    trainer.validate_model()
    return


def test_prediction(trainer, graph, molecule) -> None:

    a = trainer.predict(graph)
    b = trainer.predict(molecule)
    c = trainer.predict([graph])
    d = trainer.predict([molecule])
    with pytest.raises(TypeError):
        trainer.predict(2)
    array_equal(a, b)
    array_equal(a, c)
    array_equal(a, d)

    delattr(trainer, "_model")

    with pytest.raises(NotInitializedError):
        trainer.predict(graph)
    with pytest.raises(NotInitializedError):
        trainer.save_model_statedict()
    with pytest.raises(NotInitializedError):
        trainer.train_model(1)
    with pytest.raises(NotInitializedError):
        trainer.validate_model()
    return


def test_on_gpu(trainer) -> None:

    assert trainer._on_gpu == is_available()
