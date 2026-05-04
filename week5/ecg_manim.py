from manim import *
import numpy as np


# ---------- ECG synthesis ----------
# A single heartbeat is modeled well by a sum of Gaussians.
# Real ECG morphology has 5 named features: P, Q, R, S, T.
# R is the big spike everyone recognizes.

def gaussian(t, center, width, amplitude):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def single_beat(t, t0):
    """One heartbeat centered at t0 (seconds)."""
    p = gaussian(t, t0 - 0.20,  0.025,  0.15)
    q = gaussian(t, t0 - 0.025, 0.010, -0.15)
    r = gaussian(t, t0,         0.012,  1.20)   # the big R spike
    s = gaussian(t, t0 + 0.025, 0.010, -0.30)
    t_wave = gaussian(t, t0 + 0.20, 0.040, 0.30)
    return p + q + r + s + t_wave


def make_ecg(t, bpm=72, mains_freq=60.0, noise_amp=0.08, hum_amp=0.25, seed=42):
    """Returns (clean, noisy) signals over time array t."""
    clean, noise_component = make_ecg_components(t, bpm, mains_freq, noise_amp, hum_amp, seed)
    return clean, clean + noise_component


def make_ecg_components(t, bpm=72, mains_freq=60.0, noise_amp=0.08, hum_amp=0.25, seed=42):
    """Returns (clean, noise_component). noisy(alpha) = clean + alpha * noise_component."""
    clean, hum, broadband = make_ecg_split(t, bpm, mains_freq, noise_amp, hum_amp, seed)
    return clean, hum + broadband


def make_ecg_split(t, bpm=72, mains_freq=60.0, noise_amp=0.08, hum_amp=0.25, seed=42):
    """Returns (clean, hum, broadband) so each noise source can be animated separately."""
    period = 60.0 / bpm
    clean = np.zeros_like(t)
    n_beats = int(t[-1] / period) + 2
    for i in range(n_beats):
        clean += single_beat(t, i * period + 0.4)

    rng = np.random.default_rng(seed)
    hum = hum_amp * np.sin(2 * np.pi * mains_freq * t)
    broadband = noise_amp * rng.standard_normal(len(t))
    return clean, hum, broadband


# ---------- Scene ----------

class ColdOpen(Scene):
    def construct(self):
        # --- 1. Signal components ---
        fs = 500
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        clean, noise_component = make_ecg_components(t)

        # --- 2. Axes (invisible — raw oscilloscope vibe) ---
        axes = Axes(
            x_range=[0, duration, 1],
            y_range=[-0.6, 1.5, 0.5],
            x_length=12,
            y_length=4.5,
            tips=False,
            axis_config={"stroke_opacity": 0.0},
        ).shift(UP * 0.6)

        # --- 3. α tracker drives the trace ---
        alpha = ValueTracker(1.0)

        def trace_points(a):
            y = clean + a * noise_component
            return [axes.c2p(ti, yi) for ti, yi in zip(t, y)]

        # --- Oscilloscope-style draw: leading dot pulls the trace behind it. ---
        pts_full = trace_points(1.0)
        n_full = len(pts_full)
        draw = ValueTracker(0.0)

        pts_arr = np.array(pts_full)

        def head_position(p):
            # Smooth interpolation between samples so the dot glides instead of snapping.
            x = np.clip(p, 0.0, 1.0) * (n_full - 1)
            i = int(np.floor(x))
            f = x - i
            if i >= n_full - 1:
                return pts_arr[-1]
            return pts_arr[i] * (1 - f) + pts_arr[i + 1] * f

        trace_anim = always_redraw(
            lambda: VMobject(stroke_color=GREEN_C, stroke_width=2).set_points_as_corners(
                pts_full[: max(2, int(draw.get_value() * n_full))]
            )
        )
        head_dot = always_redraw(
            lambda: Dot(head_position(draw.get_value()), radius=0.10, color=GREEN_A)
        )

        # Faint baseline flickers in first — oscilloscope warming up.
        baseline = Line(axes.c2p(0, 0), axes.c2p(duration, 0), stroke_color=GREEN_E, stroke_width=1, stroke_opacity=0.4)
        self.play(FadeIn(baseline, run_time=0.3))

        self.add(trace_anim, head_dot)
        self.play(draw.animate.set_value(1.0), run_time=5.0, rate_func=linear)
        self.play(FadeOut(head_dot, run_time=0.25), FadeOut(baseline, run_time=0.25))

        # Freeze the drawn trace into a static mobject for the downstream swap.
        initial_trace = VMobject(stroke_color=GREEN_C, stroke_width=2).set_points_as_corners(pts_full)
        self.remove(trace_anim)
        self.add(initial_trace)
        self.wait(0.4)

        # --- 4. "Recognize this signal? Of course you do." ---
        recognize = Text("Recognize this signal?", font_size=42).to_edge(UP, buff=0.4)
        of_course = Text("Of course you do.", font_size=36, color=YELLOW).next_to(recognize, DOWN, buff=0.25)

        self.play(FadeIn(recognize, shift=DOWN * 0.2))
        self.wait(1.5)
        self.play(FadeIn(of_course, shift=DOWN * 0.2))
        self.wait(2.5)
        self.play(FadeOut(recognize), FadeOut(of_course))

        # --- 5. Swap static trace for live α-driven trace ---
        live_trace = always_redraw(
            lambda: VMobject(stroke_color=GREEN_C, stroke_width=2).set_points_as_corners(
                trace_points(alpha.get_value())
            )
        )
        self.add(live_trace)
        self.remove(initial_trace)

        # --- 6. Equation with live α value, centered at bottom. ---
        eq_left = Text("signal(t) = heart(t) +", font_size=30)
        alpha_eq = Text("1.00", font_size=30, color=YELLOW)
        dot_op = Text("·", font_size=36)
        eq_right = Text("noise(t)", font_size=30)
        equation = VGroup(eq_left, alpha_eq, dot_op, eq_right).arrange(RIGHT, buff=0.22).to_edge(DOWN, buff=0.8)
        # Lock α-symbol position so the line doesn't jitter as digits change width.
        alpha_anchor = alpha_eq.get_center()

        def update_alpha_text(m):
            new = Text(f"{alpha.get_value():.2f}", font_size=30, color=YELLOW).move_to(alpha_anchor)
            m.become(new)

        alpha_eq.add_updater(update_alpha_text)

        self.play(FadeIn(equation, shift=UP * 0.15))
        self.wait(1.0)

        # --- 7. Slowly remove the noise ---
        self.play(alpha.animate.set_value(0.0), run_time=3.0, rate_func=smooth)
        self.wait(2.5)  # let the clean heartbeat breathe

        # --- 8. Add the noise back ---
        self.play(alpha.animate.set_value(1.0), run_time=2.0, rate_func=smooth)
        self.wait(1.0)

        # Stop updaters before fading out so final frame is stable.
        alpha_eq.clear_updaters()

        # --- 9. The question ---
        question = Text("Where did this noise come from?", font_size=44).to_edge(DOWN, buff=1.6)
        self.play(
            FadeOut(equation),
            FadeIn(question, shift=UP * 0.3),
        )
        self.wait(4.0)

        self.play(FadeOut(question), FadeOut(live_trace), run_time=1.0)


def compute_fft(signal, fs):
    """One-sided magnitude spectrum."""
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(spectrum) / n
    return freqs, mag, spectrum


def notch_60hz(signal, fs, freq=60.0, bandwidth=2.0):
    """Zero out FFT bins near `freq`, return filtered time signal + spectra."""
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = np.abs(freqs - freq) < bandwidth
    filtered_spec = spectrum.copy()
    filtered_spec[mask] = 0.0
    filtered = np.fft.irfft(filtered_spec, n=n)
    return filtered, freqs, np.abs(spectrum) / n, np.abs(filtered_spec) / n


def moving_average(signal, N):
    """Symmetric N-point moving average via convolution."""
    return np.convolve(signal, np.ones(N) / N, mode="same")


def brick_lowpass(signal, fs, cutoff=40.0):
    """Zero out every FFT bin above `cutoff`. Returns (filtered, freqs, mag_before, mag_after)."""
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = freqs > cutoff
    filtered_spec = spectrum.copy()
    filtered_spec[mask] = 0.0
    filtered = np.fft.irfft(filtered_spec, n=n)
    return filtered, freqs, np.abs(spectrum) / n, np.abs(filtered_spec) / n


# ---------- Scene 2: Noise origins ----------

class NoiseOrigins(Scene):
    """
    Slow walk through where the noise actually comes from in the real world.
    Builds the equation term-by-term, layering each noise source visually.
    """
    def construct(self):
        fs = 500
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        clean, hum, broadband = make_ecg_split(t)

        # --- Title ---
        title = Text("Where does the noise come from?", font_size=40).to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(1.0)

        # Two layouts for the main wave: BIG centered, and SMALL docked left.
        big_axes = Axes(
            x_range=[0, duration, 1],
            y_range=[-0.6, 1.5, 0.5],
            x_length=11,
            y_length=3.6,
            tips=False,
            axis_config={"stroke_opacity": 0.0},
        ).move_to(DOWN * 0.2)

        small_axes = Axes(
            x_range=[0, duration, 1],
            y_range=[-0.6, 1.5, 0.5],
            x_length=5.0,
            y_length=2.0,
            tips=False,
            axis_config={"stroke_opacity": 0.0},
        ).move_to(LEFT * 3.4 + UP * 0.4)

        # Right-side dock used for whichever pure-noise component is on stage.
        noise_axes_template = lambda x_max, y_max: Axes(
            x_range=[0, x_max, x_max / 4],
            y_range=[-y_max, y_max, y_max / 2],
            x_length=5.0,
            y_length=2.0,
            tips=False,
            axis_config={"stroke_opacity": 0.0},
        ).move_to(RIGHT * 3.4 + UP * 0.4)

        def trace_on(axes, alpha_h=0.0, alpha_b=0.0):
            y = clean + alpha_h * hum + alpha_b * broadband
            pts = [axes.c2p(ti, yi) for ti, yi in zip(t, y)]
            return VMobject(stroke_color=GREEN_C, stroke_width=2).set_points_as_corners(pts)

        # --- Beat A: clean signal, big in the middle ---
        wave = trace_on(big_axes, 0, 0)
        caption = Text(
            "Start with a clean ECG: the actual heartbeat we want to see.",
            font_size=24, color=GREY_B,
        ).to_edge(DOWN, buff=0.5)

        self.play(Create(wave), run_time=1.5, rate_func=linear)
        self.play(FadeIn(caption))
        self.wait(5.0)

        # --- Beat B: pivot ---
        pivot = Text(
            "But out in the real world, the wires pick up everything around them.",
            font_size=24, color=GREY_B,
        ).move_to(caption)
        self.play(Transform(caption, pivot))
        self.wait(5.0)

        # --- Beat C: 60 Hz mains hum ---
        source_hum = Text(
            "Power lines.  Fluorescent lights.  Wall outlets.",
            font_size=26, color=YELLOW,
        ).move_to(caption)
        self.play(Transform(caption, source_hum))
        self.wait(3.5)

        # Shrink wave to the left, plus sign in the middle.
        small_clean = trace_on(small_axes, 0, 0)
        plus = Text("+", font_size=56, color=YELLOW).move_to(UP * 0.4)
        self.play(Transform(wave, small_clean), FadeIn(plus, shift=DOWN * 0.1))

        # Pure 60 Hz hum on the right.
        hum_axes = noise_axes_template(0.2, 0.4)
        t_hum = np.linspace(0, 0.2, 200)
        y_hum = 0.25 * np.sin(2 * np.pi * 60 * t_hum)
        hum_pure = VMobject(stroke_color=RED_C, stroke_width=2).set_points_as_corners(
            [hum_axes.c2p(ti, yi) for ti, yi in zip(t_hum, y_hum)]
        )
        hum_label = Text("60 Hz hum", font_size=22, color=RED_C).next_to(hum_axes, DOWN, buff=0.2)
        self.play(Create(hum_pure), FadeIn(hum_label), run_time=1.5)
        self.wait(4.0)

        # Equation grows under the wave; new term highlighted yellow.
        eq_hum = Text(
            "signal(t) = heart(t) + 0.25 · sin(2π · 60 · t)",
            font_size=28,
            t2c={"+ 0.25 · sin(2π · 60 · t)": YELLOW},
        ).move_to(caption)
        self.play(Transform(caption, eq_hum))
        self.wait(4.5)

        # Hum flies into the main wave; main wave gains the noise.
        small_with_hum = trace_on(small_axes, 1.0, 0.0)
        self.play(
            hum_pure.animate.move_to(small_axes.get_center()).scale(0.2).set_opacity(0),
            FadeOut(hum_label),
            FadeOut(plus),
            Transform(wave, small_with_hum),
            run_time=1.8,
        )
        self.wait(0.4)

        # Main wave grows back to big center, now noisy.
        big_with_hum = trace_on(big_axes, 1.0, 0.0)
        self.play(Transform(wave, big_with_hum), run_time=1.5)
        self.wait(3.5)

        # --- Beat D: broadband ---
        source_broad = Text(
            "Thermal jitter.  Electrode contact.  Tiny muscle twitches.",
            font_size=26, color=YELLOW,
        ).move_to(caption)
        self.play(Transform(caption, source_broad))
        self.wait(3.5)

        # Shrink to left again (now starting from hum-noisy state).
        small_with_hum_again = trace_on(small_axes, 1.0, 0.0)
        plus2 = Text("+", font_size=56, color=YELLOW).move_to(UP * 0.4)
        self.play(Transform(wave, small_with_hum_again), FadeIn(plus2, shift=DOWN * 0.1))

        # Pure broadband on the right.
        broad_axes = noise_axes_template(0.5, 0.3)
        rng_inset = np.random.default_rng(7)
        t_broad = np.linspace(0, 0.5, 250)
        y_broad = 0.08 * rng_inset.standard_normal(len(t_broad))
        broad_pure = VMobject(stroke_color=GREY_B, stroke_width=2).set_points_as_corners(
            [broad_axes.c2p(ti, yi) for ti, yi in zip(t_broad, y_broad)]
        )
        broad_label = Text("broadband η(t)", font_size=22, color=GREY_B).next_to(broad_axes, DOWN, buff=0.2)
        self.play(Create(broad_pure), FadeIn(broad_label), run_time=1.5)
        self.wait(4.0)

        eq_full = Text(
            "signal(t) = heart(t) + 0.25 · sin(2π · 60 · t) + 0.08 · η(t)",
            font_size=26,
            t2c={"+ 0.08 · η(t)": YELLOW},
        ).move_to(caption)
        self.play(Transform(caption, eq_full))
        self.wait(4.5)

        small_full = trace_on(small_axes, 1.0, 1.0)
        self.play(
            broad_pure.animate.move_to(small_axes.get_center()).scale(0.2).set_opacity(0),
            FadeOut(broad_label),
            FadeOut(plus2),
            Transform(wave, small_full),
            run_time=1.8,
        )
        self.wait(0.4)

        big_full = trace_on(big_axes, 1.0, 1.0)
        self.play(Transform(wave, big_full), run_time=1.5)
        self.wait(3.5)

        # --- Beat E: hand off to the cleanup question ---
        cleanup_q = Text(
            "Okay, so how do we clean up the signal?",
            font_size=30, color=YELLOW,
        ).move_to(caption)
        self.play(Transform(caption, cleanup_q))
        self.wait(5.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)


# ---------- Scene 3: Diagnose ----------

class DiagnoseNoise(Scene):
    def construct(self):
        fs = 500
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        _clean, noisy = make_ecg(t)
        freqs, mag, _ = compute_fft(noisy, fs)

        f_max = 120.0
        keep = freqs <= f_max
        freqs_v = freqs[keep]
        mag_v = mag[keep]

        # Headroom above the 60 Hz spike so labels have somewhere to live.
        y_top = max(mag_v) * 1.6

        time_axes = Axes(
            x_range=[0, duration, 1],
            y_range=[-0.6, 1.5, 0.5],
            x_length=12,
            y_length=2.7,
            tips=False,
            axis_config={"stroke_opacity": 0.4, "stroke_width": 1.5},
        )
        time_label = Text("Time domain", font_size=22)

        freq_axes = Axes(
            x_range=[0, f_max, 20],
            y_range=[0, y_top, y_top / 4],
            x_length=12,
            y_length=2.7,
            tips=False,
            axis_config={"stroke_opacity": 0.6, "stroke_width": 1.5},
        )
        freq_label = Text("Frequency domain (FFT)", font_size=22)

        # Stack the two panels evenly, then place tick / annotation children afterwards.
        time_block = VGroup(time_label, time_axes).arrange(DOWN, buff=0.15)
        freq_block = VGroup(freq_label, freq_axes).arrange(DOWN, buff=0.15)
        VGroup(time_block, freq_block).arrange(DOWN, buff=0.55).move_to(ORIGIN)

        x_axis_label = Text("Hz", font_size=18).next_to(freq_axes.x_axis, RIGHT, buff=0.2)
        freq_ticks = VGroup(*[
            Text(str(hz), font_size=16).next_to(freq_axes.c2p(hz, 0), DOWN, buff=0.12)
            for hz in [20, 40, 60, 80, 100]
        ])

        time_pts = [time_axes.c2p(ti, yi) for ti, yi in zip(t, noisy)]
        time_trace = VMobject(stroke_color=GREEN_C, stroke_width=2).set_points_as_corners(time_pts)

        spectrum_pts = [freq_axes.c2p(fi, mi) for fi, mi in zip(freqs_v, mag_v)]
        spectrum = VMobject(stroke_color=BLUE_C, stroke_width=2).set_points_as_corners(spectrum_pts)

        self.play(FadeIn(time_axes), Write(time_label))
        self.wait(1.5)
        self.play(Create(time_trace), run_time=3.5, rate_func=linear)
        self.wait(3.5)

        self.play(FadeIn(freq_axes), Write(freq_label), Write(x_axis_label), FadeIn(freq_ticks))
        self.wait(1.5)
        self.play(Create(spectrum), run_time=3.0, rate_func=linear)
        self.wait(4.0)

        # --- Annotations: labels in upper row, arrows down to peaks. ---
        heart_text = Text("heart signal", font_size=20, color=YELLOW).move_to(
            freq_axes.c2p(15, y_top * 0.88)
        )
        heart_arrow = Arrow(
            heart_text.get_corner(DL) + DOWN * 0.05,
            freq_axes.c2p(2, y_top * 0.28),
            buff=0.1, stroke_width=3, color=YELLOW,
        )

        hum_text = Text("60 Hz mains hum", font_size=20, color=RED).move_to(
            freq_axes.c2p(42, y_top * 0.88)
        )
        hum_arrow = Arrow(
            hum_text.get_corner(DR) + DOWN * 0.05,
            freq_axes.c2p(60, y_top * 0.32),
            buff=0.1, stroke_width=3, color=RED,
        )

        floor_text = Text("broadband noise", font_size=20, color=GREY_B).move_to(
            freq_axes.c2p(100, y_top * 0.88)
        )
        floor_arrow = Arrow(
            floor_text.get_bottom() + DOWN * 0.05,
            freq_axes.c2p(95, y_top * 0.1),
            buff=0.1, stroke_width=3, color=GREY_B,
        )

        self.play(GrowArrow(heart_arrow), FadeIn(heart_text))
        self.wait(4.0)
        self.play(GrowArrow(hum_arrow), FadeIn(hum_text), Indicate(spectrum, scale_factor=1.0))
        self.wait(4.0)
        self.play(GrowArrow(floor_arrow), FadeIn(floor_text))
        self.wait(5.5)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)


# ---------- Scene 3: Naive filter ----------

class NaiveFilter(Scene):
    def construct(self):
        fs = 500
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        # This scene focuses on the brick filter's behavior on 60 Hz hum specifically.
        # Drop broadband so the ringing artifact is the only thing the viewer has to track.
        # Cutoff = 20 Hz: aggressive enough to actually trim significant QRS spectrum, so
        # the Gibbs ringing is visible (~0.2 in y-units, ~10% of plot height). At 40 Hz the
        # ringing is mathematically present but invisible (~0.002).
        cutoff = 20.0
        clean_part, hum_part, _ = make_ecg_split(t)
        noisy = clean_part + hum_part
        filtered, freqs, mag_before, mag_after = brick_lowpass(noisy, fs, cutoff=cutoff)

        f_max = 120.0
        keep = freqs <= f_max
        freqs_v = freqs[keep]
        mag_b = mag_before[keep]
        mag_a = mag_after[keep]
        y_top = max(mag_b) * 1.1

        # --- Framing: easy approach? ---
        question_intro = Text("What's the easy approach?", font_size=44)
        self.play(FadeIn(question_intro, shift=DOWN * 0.2))
        self.wait(3.0)

        answer_intro = Text(
            "Heart lives below 20 Hz. Just delete everything above it.",
            font_size=28, color=YELLOW,
        ).next_to(question_intro, DOWN, buff=0.5)
        self.play(FadeIn(answer_intro))
        self.wait(4.5)

        self.play(FadeOut(question_intro), FadeOut(answer_intro))

        # --- Top: spectrum panel ---
        freq_axes = Axes(
            x_range=[0, f_max, 20],
            y_range=[0, y_top, y_top / 4],
            x_length=12,
            y_length=2.5,
            tips=False,
            axis_config={"stroke_opacity": 0.6, "stroke_width": 1.5},
        ).to_edge(UP, buff=0.6)
        freq_label = Text("Spectrum: zero everything above 20 Hz", font_size=22).next_to(freq_axes, UP, buff=0.1)
        freq_ticks = VGroup(*[
            Text(str(hz), font_size=16).next_to(freq_axes.c2p(hz, 0), DOWN, buff=0.1)
            for hz in [20, 40, 60, 80, 100]
        ])

        before_pts = [freq_axes.c2p(fi, mi) for fi, mi in zip(freqs_v, mag_b)]
        before_spec = VMobject(stroke_color=BLUE_C, stroke_width=2)
        before_spec.set_points_as_corners(before_pts)

        after_pts = [freq_axes.c2p(fi, mi) for fi, mi in zip(freqs_v, mag_a)]
        after_spec = VMobject(stroke_color=BLUE_C, stroke_width=2)
        after_spec.set_points_as_corners(after_pts)

        self.play(FadeIn(freq_axes), Write(freq_label), FadeIn(freq_ticks))
        self.play(Create(before_spec), run_time=2.0, rate_func=linear)
        self.wait(2.5)

        # Red box covering the entire region we're throwing away — everything above 40 Hz.
        spike_box = Rectangle(
            width=freq_axes.c2p(f_max, 0)[0] - freq_axes.c2p(cutoff, 0)[0],
            height=freq_axes.c2p(0, y_top)[1] - freq_axes.c2p(0, 0)[1],
            stroke_color=RED, stroke_width=3, fill_opacity=0.12, fill_color=RED,
        ).move_to(freq_axes.c2p((cutoff + f_max) / 2, y_top / 2))
        snip_label = Text("delete this band", font_size=20, color=RED).next_to(spike_box, UP, buff=0.1)

        self.play(Create(spike_box), FadeIn(snip_label))
        self.wait(2.0)
        self.play(
            Transform(before_spec, after_spec),
            spike_box.animate.stretch(0.01, 0).set_opacity(0.0),
            FadeOut(snip_label),
            run_time=1.5,
        )
        self.wait(3.0)

        # --- Bottom: time-domain payoff ---
        time_axes = Axes(
            x_range=[0, duration, 1],
            y_range=[-0.6, 1.5, 0.5],
            x_length=12,
            y_length=2.5,
            tips=False,
            axis_config={"stroke_opacity": 0.4, "stroke_width": 1.5},
        ).to_edge(DOWN, buff=0.6)
        time_label = Text("Time domain after IFFT", font_size=22).next_to(time_axes, UP, buff=0.1)

        noisy_pts = [time_axes.c2p(ti, yi) for ti, yi in zip(t, noisy)]
        noisy_trace = VMobject(stroke_color=GREEN_C, stroke_width=2)
        noisy_trace.set_points_as_corners(noisy_pts)

        filt_pts = [time_axes.c2p(ti, yi) for ti, yi in zip(t, filtered)]
        filt_trace = VMobject(stroke_color=GREEN_C, stroke_width=2)
        filt_trace.set_points_as_corners(filt_pts)

        self.play(FadeIn(time_axes), Write(time_label))
        self.play(Create(noisy_trace), run_time=2.0, rate_func=linear)
        self.wait(2.5)
        self.play(Transform(noisy_trace, filt_trace), run_time=2.0)
        self.wait(4.0)

        # --- Zoom in on one QRS to show ringing ---
        # Pick the second beat at t0 = 0.4 + 60/72 ≈ 1.233 s.
        center_t = 0.4 + 60.0 / 72.0
        zoom_window = 0.4
        mask = (t > center_t - zoom_window) & (t < center_t + zoom_window)
        zt = t[mask]
        zy = filtered[mask]
        ref_y = clean_part[mask]  # the same heartbeat with no filter, no noise

        zoom_axes = Axes(
            x_range=[zt[0], zt[-1], 0.1],
            y_range=[-0.6, 1.5, 0.5],
            x_length=8,
            y_length=4,
            tips=False,
            axis_config={"stroke_opacity": 0.0},
        ).move_to(ORIGIN)

        ref_pts = [zoom_axes.c2p(ti, yi) for ti, yi in zip(zt, ref_y)]
        ref_trace = VMobject(stroke_color=GREEN_C, stroke_width=2.5).set_points_as_corners(ref_pts)

        zoom_pts = [zoom_axes.c2p(ti, yi) for ti, yi in zip(zt, zy)]
        zoom_trace = VMobject(stroke_color=GREEN_C, stroke_width=2.5).set_points_as_corners(zoom_pts)

        zoom_title = Text("zoom: one heartbeat", font_size=24).next_to(zoom_axes, UP, buff=0.2)

        self.play(
            *[FadeOut(m) for m in [freq_axes, freq_label, freq_ticks, before_spec, time_axes, time_label, noisy_trace]],
            run_time=0.8,
        )

        # Show the clean heartbeat first so the viewer knows what a normal QRS looks like.
        self.play(FadeIn(zoom_axes), Write(zoom_title), Create(ref_trace), run_time=2.0)
        clean_label = Text("clean heartbeat (no filter)", font_size=22, color=GREY_B).next_to(zoom_axes, DOWN, buff=0.3)
        self.play(FadeIn(clean_label))
        self.wait(4.0)

        # Now apply the brick filter: ringing rides in on the QRS shoulders.
        filtered_label = Text("after the brick filter", font_size=22, color=YELLOW).next_to(zoom_axes, DOWN, buff=0.3)
        self.play(Transform(ref_trace, zoom_trace), Transform(clean_label, filtered_label), run_time=1.5)
        self.wait(3.0)

        # Highlight the two distinct phenomena in the brick-filtered output:
        #   1) Coherent ~60 Hz wiggles flanking the QRS = ringing (Gibbs-like artifact).
        #   2) Random fuzz across the whole trace = broadband (untouched by the notch).
        ring_w = zoom_axes.c2p(0.09, 0)[0] - zoom_axes.c2p(0, 0)[0]
        ring_h = zoom_axes.c2p(0, 0.45)[1] - zoom_axes.c2p(0, 0.0)[1]

        ring_left = Ellipse(
            width=ring_w, height=ring_h,
            color=RED, stroke_width=2.5, fill_opacity=0.0,
        ).move_to(zoom_axes.c2p(center_t - 0.06, 0.10))

        ring_right = Ellipse(
            width=ring_w, height=ring_h,
            color=RED, stroke_width=2.5, fill_opacity=0.0,
        ).move_to(zoom_axes.c2p(center_t + 0.06, 0.10))

        ring_text = Text("ringing artifact", font_size=22, color=RED).next_to(clean_label, DOWN, buff=0.2)

        self.play(Create(ring_left), Create(ring_right), FadeIn(ring_text))
        self.wait(5.0)

        question = Text(
            "Can we do better?",
            font_size=36, color=YELLOW,
        ).to_edge(DOWN, buff=0.4)

        self.play(
            FadeOut(clean_label),
            FadeOut(ring_text),
            FadeIn(question, shift=UP * 0.3),
        )
        self.wait(4.5)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)


# ---------- Scene 4: Moving average ----------

class MovingAverage(Scene):
    """
    Pitch a far simpler filter: average a small window of samples.
    Show the window sliding across the noisy signal as the smoothed output draws below.
    """
    def construct(self):
        fs = 500
        duration = 5.0
        N = 8
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        # Same simplification as NaiveFilter: only 60 Hz hum on top of the clean signal.
        clean_part, hum_part, _ = make_ecg_split(t)
        noisy = clean_part + hum_part
        smoothed = moving_average(noisy, N)

        # --- Title and equation ---
        title = Text("A simple filter: moving average", font_size=38).to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(2.0)

        eq = Text(
            "y[n] = (1 / N) · Σ x[n−k]",
            font_size=44,
            t2c={
                "y[n]": BLUE_C,
                "1 / N": YELLOW,
                "Σ": RED,
                "x[n−k]": GREEN_C,
            },
        ).next_to(title, DOWN, buff=0.6)
        self.play(FadeIn(eq))
        self.wait(3.0)

        # Colour-matched legend, revealed line by line so the user can narrate each part.
        leg_in = Text(
            "x[n−k] : the last N input samples",
            font_size=26,
            t2c={"x[n−k]": GREEN_C},
        ).next_to(eq, DOWN, buff=0.5)
        leg_sum = Text(
            "Σ : sum them up",
            font_size=26,
            t2c={"Σ": RED},
        ).next_to(leg_in, DOWN, buff=0.25)
        leg_avg = Text(
            "1 / N : divide to take the average",
            font_size=26,
            t2c={"1 / N": YELLOW},
        ).next_to(leg_sum, DOWN, buff=0.25)
        leg_out = Text(
            "y[n] : your new (filtered) sample",
            font_size=26,
            t2c={"y[n]": BLUE_C},
        ).next_to(leg_avg, DOWN, buff=0.25)

        self.play(FadeIn(leg_in))
        self.wait(3.0)
        self.play(FadeIn(leg_sum))
        self.wait(3.0)
        self.play(FadeIn(leg_avg))
        self.wait(3.0)
        self.play(FadeIn(leg_out))
        self.wait(4.0)

        self.play(FadeOut(eq), FadeOut(leg_in), FadeOut(leg_sum), FadeOut(leg_avg), FadeOut(leg_out))

        # --- Two stacked panels: noisy on top, smoothed below ---
        top_axes = Axes(
            x_range=[0, duration, 1],
            y_range=[-0.6, 1.5, 0.5],
            x_length=12,
            y_length=2.0,
            tips=False,
            axis_config={"stroke_opacity": 0.4, "stroke_width": 1.5},
        )
        top_label = Text("Noisy input", font_size=22)

        bot_axes = Axes(
            x_range=[0, duration, 1],
            y_range=[-0.6, 1.5, 0.5],
            x_length=12,
            y_length=2.0,
            tips=False,
            axis_config={"stroke_opacity": 0.4, "stroke_width": 1.5},
        )
        bot_label = Text("After moving average", font_size=22)

        top_block = VGroup(top_label, top_axes).arrange(DOWN, buff=0.15)
        bot_block = VGroup(bot_label, bot_axes).arrange(DOWN, buff=0.2)
        VGroup(top_block, bot_block).arrange(DOWN, buff=0.5).next_to(title, DOWN, buff=0.5)

        noisy_pts = [top_axes.c2p(ti, yi) for ti, yi in zip(t, noisy)]
        noisy_trace = VMobject(stroke_color=GREEN_C, stroke_width=2).set_points_as_corners(noisy_pts)

        smooth_pts = [bot_axes.c2p(ti, yi) for ti, yi in zip(t, smoothed)]
        smooth_trace = VMobject(stroke_color=GREEN_C, stroke_width=2.5).set_points_as_corners(smooth_pts)

        self.play(FadeIn(top_axes), Write(top_label), FadeIn(bot_axes), Write(bot_label))
        self.play(Create(noisy_trace), run_time=2.0, rate_func=linear)
        self.wait(2.5)

        # --- Sliding window across the noisy panel ---
        # Use a visually larger box than 8 samples so the viewer can actually see it.
        visual_w = 0.45
        window_h = top_axes.c2p(0, 1.5)[1] - top_axes.c2p(0, -0.6)[1]
        window = Rectangle(
            width=visual_w, height=window_h,
            stroke_color=YELLOW, stroke_width=2,
            fill_opacity=0.15, fill_color=YELLOW,
        ).move_to(top_axes.c2p(0, 0.45))

        window_tag = Text(f"window (N={N})", font_size=18, color=YELLOW)
        window_tag.add_updater(lambda m: m.next_to(window, UP, buff=0.05))
        window_tag.next_to(window, UP, buff=0.05)

        self.play(FadeIn(window), FadeIn(window_tag))
        self.wait(2.5)

        # Sweep window left-to-right while the smoothed output draws below in sync.
        self.play(
            window.animate.move_to(top_axes.c2p(duration, 0.45)),
            Create(smooth_trace),
            run_time=5.0,
            rate_func=linear,
        )
        window_tag.clear_updaters()
        self.wait(3.0)

        # --- Punchline ---
        punchline = Text("Smooth.  No ringing.", font_size=32, color=YELLOW).to_edge(DOWN, buff=0.4)
        self.play(FadeOut(window), FadeOut(window_tag), FadeIn(punchline, shift=UP * 0.2))
        self.wait(5.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)


# ---------- Scene 5: Final comparison ----------

class FinalCompare(Scene):
    """
    Side-by-side outcome: brick filter (rings) vs moving average (smooth).
    """
    def construct(self):
        fs = 500
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        # Compare both filters on the same input as the previous scenes: clean + 60 Hz hum.
        clean_part, hum_part, _ = make_ecg_split(t)
        noisy = clean_part + hum_part
        brick, _, _, _ = brick_lowpass(noisy, fs, cutoff=20.0)
        smoothed = moving_average(noisy, 8)

        title = Text("Three signals side by side", font_size=36).to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(1.5)

        # Three stacked panels: input, brick output, MA output.
        def panel(label_text, label_color):
            ax = Axes(
                x_range=[0, duration, 1],
                y_range=[-0.6, 1.5, 0.5],
                x_length=11,
                y_length=1.4,
                tips=False,
                axis_config={"stroke_opacity": 0.4, "stroke_width": 1.5},
            )
            lbl = Text(label_text, font_size=20, color=label_color)
            return VGroup(lbl, ax).arrange(DOWN, buff=0.1), ax, lbl

        in_block, in_axes, in_lbl = panel("Noisy input (clean + 60 Hz hum)", GREY_B)
        brick_block, brick_axes, brick_lbl = panel("Brick lowpass (cut above 20 Hz)", RED_C)
        ma_block, ma_axes, ma_lbl = panel("Moving average (N = 8)", GREEN_C)

        VGroup(in_block, brick_block, ma_block).arrange(DOWN, buff=0.35).next_to(title, DOWN, buff=0.35)

        in_trace = VMobject(stroke_color=GREEN_C, stroke_width=2).set_points_as_corners(
            [in_axes.c2p(ti, yi) for ti, yi in zip(t, noisy)]
        )
        brick_trace = VMobject(stroke_color=GREEN_C, stroke_width=2).set_points_as_corners(
            [brick_axes.c2p(ti, yi) for ti, yi in zip(t, brick)]
        )
        ma_trace = VMobject(stroke_color=GREEN_C, stroke_width=2).set_points_as_corners(
            [ma_axes.c2p(ti, yi) for ti, yi in zip(t, smoothed)]
        )

        # --- 1. Noisy input ---
        self.play(FadeIn(in_axes), Write(in_lbl))
        self.play(Create(in_trace), run_time=2.0, rate_func=linear)
        self.wait(3.5)

        # --- 2. Brick output ---
        self.play(FadeIn(brick_axes), Write(brick_lbl))
        self.play(Create(brick_trace), run_time=2.0, rate_func=linear)
        ring_caption = Text("ringing", font_size=18, color=RED).move_to(brick_axes.c2p(0.7, 1.1))
        self.play(FadeIn(ring_caption))
        self.wait(4.5)

        # --- 3. Moving average output ---
        self.play(FadeIn(ma_axes), Write(ma_lbl))
        self.play(Create(ma_trace), run_time=2.0, rate_func=linear)
        smooth_caption = Text("clean and smooth", font_size=18, color=YELLOW).move_to(ma_axes.c2p(0.9, 1.1))
        self.play(FadeIn(smooth_caption))
        self.wait(3.5)

        # --- All three on screen — long pause for narration ---
        self.wait(12.0)

        # --- Closing beat ---
        closing = Text("Different filters, different tradeoffs.", font_size=32, color=YELLOW)
        self.play(
            *[FadeOut(m) for m in [in_axes, in_lbl, in_trace,
                                    brick_axes, brick_lbl, brick_trace, ring_caption,
                                    ma_axes, ma_lbl, ma_trace, smooth_caption]],
            run_time=1.0,
        )
        self.play(FadeIn(closing, shift=UP * 0.2))
        self.wait(5.0)
        self.play(FadeOut(title), FadeOut(closing), run_time=1.0)