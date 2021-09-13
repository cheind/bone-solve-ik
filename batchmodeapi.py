body = BodyBuilder.finalize()

body.logdof_shape  # (Ex6) each edge rx,ry,rz

logdofs = torch.zeros((B,) + body.logdof_shape)
logdofsmask = ...  # (1, + body.logdof_shape)

body.create_logdofs(batch_size)  # ExBx6
body.create_logaffine()  # Ex3x3

dofs.log2angle(logdofs, affines)  # BxEx3
dofs.angle2log(angles, invaffines)

dofs.log2exp()  # BxEx2 cos(theta), sin(theta)

kinematics.fk(body, logdofs)


logrots = net(anchors)  # (BxEx3x2)

constraints = body.constraints  # (Ex3x3)
rot.exp(
    logrots, constraints
)  # first computes z (cos(theta),sin(theta)) then matrices (BxEx3x4x4)


body = ...
motion = body.motion_data()

motion.body_world
motion.uconstrained_rotations  # (N*3, 2)
motion.reset()
motion.get_parameters()
motion.set_rotation("a", "b", [rx, ry, rz])
motion.set_translation("root", "torso", [...])
motion.uniform_sample(n)
motion.normal_sample(n)


fk(body, motion)
