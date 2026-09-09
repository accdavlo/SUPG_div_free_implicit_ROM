from .dec import DeC
from .dec_step import DeC_one_step, DeC_one_step_MOR
from .sources_residuals import (
    define_sources, define_sources_MOR, define_sources_MOR_uv_p,
    define_residuals, define_residuals_MOR, define_residuals_MOR_uv_p,
    define_GF_residuals, define_GF_residuals_MOR, define_GF_residuals_MOR_uv_p,
)
from .stabilization import (
    SUPG_stabilization, SUPG_stabilization_MOR, SUPG_stabilization_MOR_uv_p,
    SUPG_GF_stabilization, SUPG_GF_stabilization_MOR, SUPG_GF_stabilization_MOR_uv_p,
    OSS_stabilization, OSS_GF_stabilization,
    OSS_curl_stabilization, OSS_GF_curl_stabilization,
)