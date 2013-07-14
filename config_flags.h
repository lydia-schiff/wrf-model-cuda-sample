#ifndef CONFIG_FLAGS_H
#define CONFIG_FLAGS_H

typedef struct _config_flags {
	int h_sca_adv_order;
	int h_mom_adv_order;
	int map_proj;
	int nested;
	int open_xe;
	int open_xs;
	int open_ye;
	int open_ys;
	int periodic_x;
	int periodic_y;
	int polar;
	int specified;
	int symmetric_xe;
	int symmetric_xs;
	int symmetric_ye;
	int symmetric_ys;
	int v_sca_adv_order;
	int v_mom_adv_order;
	int fft_filter_lat;
	int w_damping;
	int bl_pbl_physics;
	int cu_physics;
	int grid_fdda;
	int ra_lw_physics;
	int ra_sw_physics;
	int shcu_physics;
} config_flags;



#endif

