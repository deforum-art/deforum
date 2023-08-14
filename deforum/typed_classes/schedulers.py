from enum import Enum

from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    DDIMParallelScheduler,
    DDPMParallelScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    RePaintScheduler,
    ScoreSdeVeScheduler,
    ScoreSdeVpScheduler,
    UnCLIPScheduler,
    UniPCMultistepScheduler,
    VQDiffusionScheduler,
    LMSDiscreteScheduler,
    DPMSolverSDEScheduler,
)

class SchedulerType(Enum):
    """
    An enumeration representing different scheduler types.
    
    Enum Members
    ------------
    EULER_ANCESTRAL, EULER, PNDM, DPMPP_SINGLESTEP, DPMPP_MULTISTEP,
    LMS, DDIM, UNIPC, SDE,
    DDIM_INVERSE, DDIM_PARALLEL, DDPM_PARALLEL, DEIS_MULTISTEP, 
    DPMPP_MULTISTEP_INVERSE, HEUN_DISCRETE,
    IPNDM, K_DPM2_ANC, K_DPM2_KARRASVE, PAINT_REPAINT, SCORESDEVE, 
    SCORESDEVP, UNCLIP, VQ_DIFF

    Methods
    -------
    to_scheduler :
        returns a dictionary mapping enum members to their respective scheduler classes
    """

    DDIM = "ddim"
    DDPM = "ddpm"
    DEIS = "deis"
    DPMS = "dpms"
    DPMM = "dpmm"
    HEUN = "heun"
    KDPM2 = "kdpm2"
    KDPM2_A = "kdpm2_a"
    LMS = "lms"
    PNDM = "pndm"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    UNIPC = "unipc"
    DPMPP_SINGLESTEP = "dpmpp_singlestep"
    DPMPP_MULTISTEP = "dpmpp_multistep"

    def to_scheduler(self):
        """
        Maps the enum's values to the corresponding scheduler objects.
        Returns
        -------
        object
            The scheduler object corresponding to the enum value.
        """
        return {
            self.DDIM.value: DDIMScheduler,
            self.DDPM.value: DDPMScheduler,
            self.DEIS.value: DEISMultistepScheduler,
            self.DPMS.value: DPMSolverSinglestepScheduler,
            self.DPMM.value: DPMSolverMultistepScheduler,
            self.HEUN.value: HeunDiscreteScheduler,
            self.KDPM2_A.value: KDPM2AncestralDiscreteScheduler,
            self.KDPM2.value: KDPM2DiscreteScheduler,
            self.LMS.value: LMSDiscreteScheduler,
            self.PNDM.value: PNDMScheduler,
            self.EULER.value: EulerDiscreteScheduler,
            self.EULER_ANCESTRAL.value: EulerAncestralDiscreteScheduler,
            self.UNIPC.value: UniPCMultistepScheduler,
            self.DPMPP_SINGLESTEP.value: DDIMParallelScheduler,
            self.DPMPP_MULTISTEP.value: DDPMParallelScheduler,

        }.get(self.value, EulerAncestralDiscreteScheduler)