from isaaclab._stubcfg import Cfg


class UsdFileCfg(Cfg): pass
class RigidBodyPropertiesCfg(Cfg): pass
class MassPropertiesCfg(Cfg): pass
class CollisionPropertiesCfg(Cfg): pass
class ArticulationRootPropertiesCfg(Cfg): pass
class DomeLightCfg(Cfg): pass
class PhysxCfg(Cfg): pass
class SimulationCfg(Cfg): pass


class SimulationContext:
    @staticmethod
    def instance():
        from isaaclab._fake_runtime import STATE
        return STATE["sim"]
