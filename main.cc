#include <chrono>
#include "spade.h"
#include "proto/hywall_interop.h"

#include "typedef.h"
#include "calc_u_bulk.h"

#include "PTL.h"

static inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void set_channel_slip(auto& prims)
{
    const real_t t_wall = 0.1;
    const auto& grid = prims.get_grid();
    for (auto lb: range(0, grid.get_num_local_blocks()))
    {
        const auto& lb_glob = grid.get_partition().get_global_block(lb);
        int idc = 0;
        for (int dir = 2; dir <= 3; ++dir)
        {
            const auto& idomain = grid.is_domain_boundary(lb_glob);
            if (idomain(dir/2, dir%2))
            {
                const auto lb_idx = spade::ctrs::expand_index(lb_glob, grid.get_num_blocks());
                const auto nvec_out = v3i(0,2*idc-1,0);
                const int j = idc*(grid.get_num_cells(1)-1);
                auto r1 = range(-grid.get_num_exchange(0), grid.get_num_cells(0) + grid.get_num_exchange(0));
                auto r2 = range(-grid.get_num_exchange(2), grid.get_num_cells(2) + grid.get_num_exchange(2));
                for (auto ii: r1*r2)
                {
                    for (int nnn = 0; nnn < 2; ++nnn)
                    {
                        spade::grid::cell_idx_t i_d(ii[0], j-(nnn+0)*nvec_out[1], ii[1], lb);
                        spade::grid::cell_idx_t i_g(ii[0], j+(nnn+1)*nvec_out[1], ii[1], lb);
                        prim_t q_d, q_g;
                        for (auto n: range(0,5)) q_d[n] = prims(n, i_d[0], i_d[1], i_d[2], i_d[3]);
                        const auto x_g = grid.get_comp_coords(i_g);
                        const auto x_d = grid.get_comp_coords(i_d);
                        const auto n_g = calc_normal_vector(grid.coord_sys(), x_g, i_g, 1);
                        const auto n_d = calc_normal_vector(grid.coord_sys(), x_d, i_d, 1);
                        q_g.p()   =  q_d.p();
                        q_g.T()   =  q_d.T();
                        q_g.u()   =  q_d.u();
                        q_g.v()   = -q_d.v()*n_d[1]/n_g[1];
                        q_g.w()   =  q_d.w();
                        for (auto n: range(0,5)) prims(n, i_g[0], i_g[1], i_g[2], i_g[3]) = q_g[n];
                    }
                }
            }
            ++idc;
        }
    }
}

int main(int argc, char** argv)
{
    spade::parallel::mpi_t group(&argc, &argv);
    const std::size_t dim = 3;

    std::string input_filename = "none";
    for (auto i: range(0, argc))
    {
        std::string line(argv[i]);
        if (ends_with(line, ".ptl"))
        {
            input_filename = line;
            if (group.isroot()) print("Reading", input_filename);
        }
    }
    if (input_filename == "none")
    {
        if (group.isroot()) print("E: No input file name provided!");
        return 1;
    }
    
    PTL::PropertyTree input;
    input.Read(input_filename);
    
    std::vector<int>    nblk     = input["Grid"]["num_blocks"];
    std::vector<int>    ncell    = input["Grid"]["num_cells"];
    std::vector<int>    nexg     = input["Grid"]["num_exchg"];
    std::vector<real_t> bbox     = input["Grid"]["dims"];
    
    real_t     targ_cfl         = input["Time"]["cfl"];
    int        nt_max           = input["Time"]["nt_max"];
    int        nt_skip          = input["Time"]["nt_skip"];
    int        checkpoint_skip  = input["Time"]["ck_skip"];
    bool       output_timing    = input["Time"]["output_timing"];
    
    real_t                Twall    = input["WallModel"]["wallTemperature"];
    real_t                prandtl  = input["WallModel"]["fluidPrandtl"];
    real_t                mu_wall  = input["Fluid"]["mu_wall"];
    real_t                tau_wall = input["Fluid"]["tau_wall"];
    real_t                rho_b    = input["Fluid"]["rho_b"];
    real_t                rgas     = input["WallModel"]["gasConstant"];
    real_t                fluidCp  = input["WallModel"]["fluidCp"];
    
    
    real_t                eps_p  = input["Num"]["eps_p"];
    real_t                eps_T  = input["Num"]["eps_T"];
    bool              wm_enable  = input["Num"]["wm_enable"];

    std::string        init_file = input["IO"]["init_file"];
    std::string        out_dir   = input["IO"]["out_dir"];
    
    spade::coords::identity<real_t> coords;
    
    std::filesystem::path primary(out_dir);
    if (!std::filesystem::is_directory(primary)) std::filesystem::create_directory(primary);
    
    std::filesystem::path ck_dir(primary / "checkpoint");
    if (!std::filesystem::is_directory(ck_dir)) std::filesystem::create_directory(ck_dir);

    const std::string data_out = primary / "viz";
    
    spade::ctrs::array<int, dim> num_blocks;
    spade::ctrs::array<int, dim> cells_in_block;
    spade::ctrs::array<int, dim> exchange_cells;
    spade::bound_box_t<real_t, dim> bounds;
    
    for (auto i: range(0, dim))
    {
        num_blocks[i]     = nblk[i];
        cells_in_block[i] = ncell[i];
        exchange_cells[i] = nexg[i];
        bounds.min(i)     = bbox[2*i];
        bounds.max(i)     = bbox[2*i + 1];
    }
    
    spade::grid::cartesian_grid_t grid(num_blocks, cells_in_block, exchange_cells, bounds, coords, group);
    
    real_t delta = 0.5*(bounds.size(1));
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;
    
    spade::grid::grid_array prim (grid, fill1);
    spade::grid::grid_array rhs (grid, fill2);
    
    spade::fluid_state::ideal_gas_t<real_t> air;
    air.R = rgas;
    air.gamma = fluidCp/(fluidCp-rgas);

    spade::viscous_laws::power_law_t visc_law(mu_wall, Twall, 0.76, prandtl, air);
    
    const real_t Lx  = bounds.size(0);
    const real_t Lz  = bounds.size(2);

    real_t Tref = 7.0*Twall;
    real_t p0 = rho_b*air.get_R()*Tref;
    real_t aw = std::sqrt(air.get_gamma()*air.get_R()*Twall);
    real_t u0 = 800*std::sqrt(tau_wall*rho_b);

    const int nmode = 11;
    using point_type = decltype(grid)::coord_point_type;
    auto ini = [&](const point_type& x) -> prim_t
    {
        real_t yh = x[1]/delta;
        prim_t output;
        output.p() = p0;
        output.T() = Tref - (Tref - Twall)*yh*yh;
        output.u() = (3.0/2.0)*u0*(1.0-yh*yh);
        output.v() = 0;
        output.w() = 0;

        real_t up = 0.0;
        real_t vp = 0.0;
        real_t wp = 0.0;
        int imin = 1;
        for (int ii = imin; ii < imin + nmode; ++ii)
        {
            real_t ampl = 0.1*u0*(1.0-yh*yh)/(ii*ii);
            real_t freq_x = 2.0*spade::consts::pi*ii/(0.5*bounds.max(0));
            real_t freq_y = 2.0*spade::consts::pi*ii/(0.5*delta);
            real_t freq_z = 2.0*spade::consts::pi*ii/(0.5*bounds.max(2));
            real_t phase_x = std::sin(14*ii)*2.0*spade::consts::pi;
            real_t phase_y = std::sin(10*ii)*2.0*spade::consts::pi;
            real_t phase_z = std::sin(17*ii)*2.0*spade::consts::pi;
            up += ampl*std::sin(freq_x*x[0]-phase_x)*std::sin(freq_y*x[1]+phase_x)*std::sin(freq_z*x[2]-phase_x);
            vp += ampl*std::sin(freq_x*x[0]+phase_y)*std::sin(freq_y*x[1]-phase_y)*std::sin(freq_z*x[2]+phase_y);
            wp += ampl*std::sin(freq_x*x[0]-phase_z)*std::sin(freq_y*x[1]+phase_z)*std::sin(freq_z*x[2]-phase_z);
        }

        output.u() += up;
        output.v() += vp;
        output.w() += wp;
        
        return output;
    };
    
    spade::algs::fill_array(prim, ini);

    real_t ub, rhob;
    calc_u_bulk(prim, air, ub, rhob);
    real_t ratio = rho_b/rhob;

    auto rhob_correct = [&](const prim_t& val)
    {
        cons_t w;
        prim_t q;
        spade::fluid_state::convert_state(val, w, air);
        w.rho() = ratio*w.rho();
        spade::fluid_state::convert_state(w, q, air);
        return q;
    };
    spade::algs::transform_inplace(prim, rhob_correct);
    calc_u_bulk(prim, air, ub, rhob);
    if (group.isroot())
    {
        print("Corrected rho_b with ratio", ratio);
        print("Specified: ", rho_b);
        print("Calculated:", rhob);
    }
    
    if (init_file != "none")
    {
        if (group.isroot()) print("reading...");
        spade::io::binary_read(init_file, prim);
        if (group.isroot()) print("Init done.");
        grid.exchange_array(prim);
        set_channel_slip(prim);
    }
    
    spade::convective::totani_lr        tscheme(air);
    spade::convective::pressure_diss_lr dscheme(air, eps_p, eps_T);
    spade::viscous::visc_lr             visc_scheme(visc_law);
    

    auto get_u = [&](const prim_t& val){return std::sqrt(air.gamma*air.R*val.T()) + std::sqrt(val.u()*val.u() + val.v()*val.v() + val.w()*val.w());};

    spade::reduce_ops::reduce_max<real_t> max_op;
    
    
    
    const real_t dx       = spade::utils::min(grid.get_dx(0), grid.get_dx(1), grid.get_dx(2));
    const real_t umax_ini = spade::algs::transform_reduce(prim, get_u, max_op);
    const real_t dt       = targ_cfl*dx/umax_ini;
    
    const real_t force_term = tau_wall/(delta*rho_b);
    auto source = [&](const prim_t& val)
    {
        cons_t w;
        spade::fluid_state::convert_state(val, w, air);
        flux_t output;
        output.continuity() = 0.0;
        output.energy()     = w.rho()*force_term*val.u();
        output.x_momentum() = w.rho()*force_term;
        output.y_momentum() = 0.0;
        output.z_momentum() = 0.0;
        return output;
    };
    
    spade::bound_box_t<bool, grid.dim()> boundary = true;
    boundary.min(1) = false;
    boundary.max(1) = false;
    
    spade::proto::hywall_binding_t wall_model(prim, rhs, air);
    wall_model.read(input["WallModel"]);
    for (auto& b: boundary) b = !b;
    wall_model.init(prim, boundary);
    for (auto& b: boundary) b = !b;
    wall_model.set_dt(dt);

    auto boundary_cond = [&](auto& q, const auto& t)
    {
        grid.exchange_array(q);
        set_channel_slip(q);
    };

    auto calc_rhs = [&](auto& rhsin, const auto& qin, const auto& tin) -> void
    {
        rhsin = 0.0;
        spade::pde_algs::flux_div(qin, rhsin, tscheme, dscheme);
        auto policy = spade::pde_algs::block_flux_all;
        spade::pde_algs::flux_div(qin, rhsin, policy, boundary, visc_scheme);
        wall_model.sample(qin, visc_law);
        wall_model.solve();
        wall_model.apply_flux(rhsin);
        spade::pde_algs::source_term(qin, rhsin, source);
    };
    
    cons_t transform_state;
    spade::fluid_state::state_transform_t trans(transform_state, air);

    real_t time0 = 0.0;
    spade::time_integration::time_axis_t       axis    (time0, dt);
    spade::time_integration::ssprk3_t          alg;
    spade::time_integration::integrator_data_t q       (prim, rhs, alg);
    spade::time_integration::integrator_t      time_int(axis, alg, q, calc_rhs, boundary_cond, trans);
    
    spade::timing::mtimer_t tmr("advance");
    std::ofstream myfile("hist.dat");
    for (auto nt: range(0, nt_max+1))
    {
        const real_t umax   = spade::algs::transform_reduce(time_int.solution(), get_u, max_op);
        calc_u_bulk(time_int.solution(), air, ub, rhob);
        
        if (group.isroot())
        {
            const real_t cfl = umax*dt/dx;
            const int pn = 10;
            print(
                "nt: ", spade::utils::pad_str(nt,   pn),
                "cfl:", spade::utils::pad_str(cfl,  pn),
                "u+a:", spade::utils::pad_str(umax, pn),
                "mb: ", spade::utils::pad_str(ub/aw,pn),
                "rb: ", spade::utils::pad_str(rhob, pn),
                "dx: ", spade::utils::pad_str(dx,   pn),
                "dt: ", spade::utils::pad_str(dt,   pn)
            );
            myfile << nt << " " << cfl << " " << umax << " " << ub << " " << rhob << " " << dx << " " << dt << std::endl;
            myfile.flush();
        }
        if (nt%nt_skip == 0)
        {
            if (group.isroot()) print("Output solution...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "prims"+nstr;
            spade::io::output_vtk(data_out, filename, grid, time_int.solution());
            if (group.isroot()) print("Done.");
        }
        if (nt%checkpoint_skip == 0)
        {
            if (group.isroot()) print("Output checkpoint...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "check"+nstr;
            filename = std::string(ck_dir)+"/"+filename+".bin";
            spade::io::binary_write(filename, time_int.solution());
            if (group.isroot()) print("Done.");
        }
        tmr.start("advance");
        time_int.advance();
        tmr.stop ("advance");
        if (group.isroot()) print(tmr);
        if (std::isnan(umax))
        {
            if (group.isroot())
            {
                print("A tragedy has occurred!");
            }
            group.sync();
            return 155;
        }
    }
    return 0;
}
