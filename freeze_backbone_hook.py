from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    def __init__(self, freeze_epochs=20):
        self.freeze_epochs = freeze_epochs
        self.frozen = False
    
    def before_train_epoch(self, runner):
        """epoch 시작 전에 호출"""
        current_epoch = runner.epoch

        if current_epoch < self.freeze_epochs and not self.frozen:
            self._freeze_backbone(runner)
            self.frozen = True
            runner.logger.info(f'Epoch{current_epoch}: Backbone frozen. Training Head only.')
        
        elif current_epoch >= self.freeze_epochs and self.frozen:
            self._unfreeze_backbone(runner)
            self.frozen = False
            runner.logger.info(f'Epoch{current_epoch}: Backbone unfrozen. Training all layers.')
    
    def _freeze_backbone(self, runner):
        """백본의 모든 파라미터를 동결"""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        if hasattr(model, 'backbone'):
            frozen_params = 0
            for param in model.backbone.parameters():
                param.requires_grad = False
                frozen_params+=param.numel()
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            runner.logger.info(f'  Frozen Params: {frozen_params:,}')
            runner.logger.info(f'  Trainable Params: {trainable_params:,}')
        else:
            runner.logger.warning('Warning: No backbone found in model')

    
    def _unfreeze_backbone(self, runner):
        """백본의 모든 파라미터 해제"""
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = True
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            runner.logger.info(f'  Trainable Params: {trainable_params:,}')