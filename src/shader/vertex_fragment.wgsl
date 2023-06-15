// Vertex shader

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32
) -> @builtin(position) vec4f {
    let v_id = vertex_id % 6u;
    switch v_id {
        case 0u, 5u: {
            return vec4(vec2(-1.0), 0.0, 1.0);
        }
        case 1u: {
            return vec4(1.0, -1.0, 0.0, 1.0);
        }
        case 2u, 3u: {
            return vec4(vec2(1.0), 0.0, 1.0);
        }
        default: {
            return vec4(-1.0, 1.0, 0.0, 1.0);
        }
    }
}

// Fragment shader

fn xyz_lab_finv(t: f32) -> f32 {
    let delta = 6. / 29.;
    if t > delta {
        return pow(t, 3.);
    } else {
        return 3. * pow(delta, 2.) * (t - 4. / 9.);
    }
}

fn lab_to_xyz(lab: vec3f) -> vec3f {
    let D65 = vec3(0.95047, 1., 1.08883);
    let l = (lab.x + 0.16) / 1.16;
    return D65 * vec3(
        xyz_lab_finv(l + lab.y / 5.),
        xyz_lab_finv(l),
        xyz_lab_finv(l - lab.z / 2.),
    );
}

fn xyz_to_rgb(color: vec3f) -> vec3f {
    let tx = mat3x3(
        3.2404542,
        -0.9692660,
        0.0556434,
        -1.5371385,
        1.8760108,
        -0.2040259,
        -0.4985314,
        0.0415560,
        1.0572252,
    );
    return tx * color;
}

struct SampleCommand {
    color: vec4f,
    center: vec2f,
    radius: f32,
    fuzz: f32,
};

@group(0) @binding(0)
var<storage> samples: array<SampleCommand>;

fn sample_eval(sample: SampleCommand, pos: vec2f) -> vec4f {
    let dist = pos - sample.center;
    let alpha = 1. - smoothstep(
        sample.radius * (1. - 0.2 * sample.fuzz) - 1.,
        sample.radius * (1. + 0.2 * sample.fuzz) + 1.,
        length(dist)
    );
    return vec4(
        sample.color.xyz, // lab
        sample.color.a * alpha
    );
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    var color = vec4(vec3(0.), 1.);
    for (var i = 0u; i < arrayLength(&samples); i++) {
        let sample = samples[i];
        let sample_color = sample_eval(sample, pos.xy);
        let alpha = sample_color.a + color.a * (1. - sample_color.a);
        color = vec4(
            mat2x3(sample_color.xyz, color.xyz) * (vec2(sample_color.a, color.a * (1. - sample_color.a)) / alpha),
            alpha
        );
    }
    return vec4(xyz_to_rgb(lab_to_xyz(color.xyz)), 1.);
}
