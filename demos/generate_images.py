#coding=utf8

################################################################################

import math
import os
import dolfin

import myVTKPythonLibrary as myvtk
import dolfin_mech        as dmech
import dolfin_warp        as dwarp

################################################################################

def generate_images_and_meshes_from_RivlinCube(
        images_folder    : str   = "generate_images",
        images_n_dim     : int   = 2,
        images_n_voxels  : int   = 100,
        mesh_size        : float = None,
        deformation_type : str   = "compx",
        texture_type     : str   = "tagging",
        noise_level      : float = 0,
        k_run            : int   = None,
        run_model        : bool  = True,
        generate_images  : bool  = True             ):

    if not os.path.exists(images_folder):
        os.mkdir(images_folder)

    images_L = 1.

    if   (images_n_dim == 2):
        working_basename = "square"
    elif (images_n_dim == 3):
        working_basename = "cube"
    working_basename += "-"+deformation_type
    if (mesh_size is not None):
        working_basename += "-h="+str(mesh_size)

    # print (run_model)
    if (run_model):
        if (mesh_size is None):
            mesh_size = images_L/images_n_voxels
        cube_params = {"X0":0.2, "Y0":0.2, "X1":0.8, "Y1":0.8, "l":mesh_size, "mesh_filebasename":images_folder+"/"+working_basename+"-mesh"}
        if (images_n_dim == 3):
            cube_params["Z0"] = 0.2
            cube_params["Z1"] = 0.8

        mat_params = {"model":"CGNH", "parameters":{"E":1., "nu":0.3}}

        step_params = {"dt_ini":1/20}

        const_params = {"type":"blox"}
        if (deformation_type == "compx"):
            load_params = {"type":"pres", "f":0.3}
        elif (deformation_type == "grav"):
            load_params = {"type":"volu", "f":0.3}

        #### writing files
        mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.run_RivlinCube_Mesh(dim=images_n_dim, params=cube_params)
        dolfin.File(images_folder+"/"+working_basename+"-meshcoarse.xml") << mesh
        mesh=dolfin.refine(mesh)
        dolfin.File(images_folder+"/"+working_basename+"-meshrefined.xml") << mesh

        dmech.run_RivlinCube_Hyperelasticity(
            dim                                    = images_n_dim,
            cube_params                            = cube_params,
            mat_params                             = mat_params,
            step_params                            = step_params,
            const_params                           = const_params,
            load_params                            = load_params,
            res_basename                           = images_folder+"/"+working_basename,
            write_vtus_with_preserved_connectivity = True,
            verbose                                = 1                                 )

    if (generate_images):
        ref_image = myvtk.createImageFromSizeAndRes(
            dim  = images_n_dim,
            size = images_L,
            res  = images_n_voxels)

        if (texture_type == "tagging"):
            s = [0.1]*images_n_dim
            ref_image_model = lambda X:math.sqrt(abs(math.sin(math.pi*X[0]/s[0]))
                                               * abs(math.sin(math.pi*X[1]/s[1])))

        if (noise_level == 0):
            noise_params = {"type":"no"}
        else:
            noise_params = {"type":"normal", "stdev":noise_level}

        dwarp.compute_warped_images(
            working_folder                  = images_folder           ,
            working_basename                = working_basename        ,
            working_ext                     = "vtu"                   ,
            working_displacement_field_name = "U"                     ,
            ref_image                       = ref_image               ,
            ref_frame                       = 0                       ,
            ref_image_model                 = ref_image_model         ,
            noise_params                    = noise_params            ,
            suffix                          = texture_type+"-noise="+str(noise_level)+(k_run is not None)*("-run="+str(k_run).zfill(2)),
            print_warped_mesh               = 0                       ,
            verbose                         = 0                                                                                        )

########################################################################

if (__name__ == "__main__"):
    import fire
    fire.Fire(generate_images_and_meshes_from_RivlinCube)