import pandas as pd
from bentoml import BentoService, api, artifacts, env
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact("model")])
class HousePredictor(BentoService):
    """
    A miniumum service exposing house prediction
    """

    @api(input=DataframeInput(), route="api/v1/predict", batch=True)
    def predict(self, df: pd.DataFrame):
        """
        Inference api for predicting house prices
        """

        return self.artifacts.model.predict(df)
