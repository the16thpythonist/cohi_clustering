from cohi_clustering.models import ContrastiveCNN


class TestContrastiveCNN:
    
    def test_construction_basically_works(self):
        
        model = ContrastiveCNN(
            input_shape=(1, 28, 28),
            resnet_units=64,
            embedding_units=[128, 256],
            projection_units=[256, 1024],
            contrastive_factor=1.0,
        )
        assert isinstance(model, ContrastiveCNN)
        
    def test_forward_basically_works(self):
        pass
        