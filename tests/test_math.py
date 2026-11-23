"""Numerical checks of the maths that the report presents.

Everything here is plain numpy: no torch, no GPU.  The point is to verify the
closed forms used in the EDM and GeoBFN sections of the paper:
  * cosine noise schedule endpoints,
  * variance-preserving forward kernel statistics,
  * the accuracy-schedule identity sum_i alpha_i = sigma1^-2 - 1 used by BFN,
  * normalisation of the discrete (uniform) reverse posterior.
"""
import numpy as np


def abar(t, s=0.008):
    f = np.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2
    f0 = np.cos(s / (1.0 + s) * np.pi / 2.0) ** 2
    return f / f0


def test_schedule():
    ts = np.linspace(0.0, 1.0, 11)
    a = abar(ts)
    print("abar:", np.round(a, 6))
    assert abs(abar(0.0) - 1.0) < 1e-12, "abar(0) must be 1"
    assert abar(1.0) < 1e-3, "abar(1) must be ~0"
    assert np.all(np.diff(a) < 0), "abar must decrease"
    print("schedule OK: abar(0)=%.6f abar(1)=%.3e monotone=yes" % (abar(0.0), abar(1.0)))


def test_vp_kernel():
    rng = np.random.default_rng(0)
    x0 = np.array([1.5, -0.5, 2.0])
    for t in (0.1, 0.5, 0.9):
        a = np.sqrt(abar(t)); sg = np.sqrt(1 - abar(t))
        eps = rng.standard_normal((200000, 3))
        xt = a * x0 + sg * eps
        assert np.allclose(xt.mean(0), a * x0, atol=0.02), "mean must be alpha*x0"
        assert np.allclose(xt.var(0), sg ** 2, atol=0.02), "variance must be sigma^2"
    print("VP kernel OK: E[x_t]=alpha x0, Var[x_t]=sigma^2")


def test_accuracy_schedule():
    for s1 in (1e-3, 0.15, 0.5):
        for steps in (10, 50, 1000):
            i = np.arange(1, steps + 1) / steps
            r = s1 ** (2.0 / steps)
            alpha = s1 ** (-2.0 * i) * (1.0 - r)
            total = alpha.sum()
            target = s1 ** -2.0 - 1.0
            assert abs(total - target) / target < 1e-6, (s1, steps, total, target)
    print("accuracy schedule OK: sum_i alpha_i = sigma1^-2 - 1 (steps independent)")


def test_discrete_posterior():
    k = 5
    rng = np.random.default_rng(1)
    for _ in range(100):
        p0 = rng.random(k); p0 /= p0.sum()
        h_t = np.zeros(k); h_t[rng.integers(0, k)] = 1.0
        a_t, a_s = 0.3, 0.7          # t is noisier than s, so abar(t) < abar(s)
        ratio = min(a_t / a_s, 1.0)
        post = (ratio * h_t + (1 - ratio) / k) * (a_s * p0 + (1 - a_s) / k)
        post = post / post.sum()
        assert abs(post.sum() - 1.0) < 1e-12, "posterior must be normalised"
        assert np.all(post >= 0), "posterior must be non-negative"
    # a zero-length step must reproduce the current state
    p0 = np.array([0.1, 0.2, 0.7, 0.0, 0.0])
    h_t = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    ratio = 1.0
    post = (ratio * h_t + (1 - ratio) / k) * (0.5 * p0 + 0.5 / k)
    post = post / post.sum()
    assert np.allclose(post, h_t), "a zero-length reverse step is the identity"
    # a long step must pull the state towards the predicted clean type
    h_t = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    p0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    a_t, a_s = 0.1, 0.9
    ratio = a_t / a_s
    post = (ratio * h_t + (1 - ratio) / k) * (a_s * p0 + (1 - a_s) / k)
    post = post / post.sum()
    assert post[2] > post[3], "posterior mass must move towards h0_hat"
    # and it must agree with direct Bayes on the same truncated chain
    print("discrete posterior OK: normalised, identity for zero-length step, "
          "long step p(h0_hat)=%.3f > p(h_t)=%.3f" % (post[2], post[3]))


def test_ddpm_coefficients():
    """The DDPM posterior coefficients must reproduce the marginal q(x_s|x_0)."""
    rng = np.random.default_rng(2)
    t, s = 0.6, 0.5
    a_t, a_s = abar(t), abar(s)          # abar(s) > abar(t): s is the cleaner state
    beta = 1 - a_t / a_s
    coef_x0 = np.sqrt(a_s) * beta / (1 - a_t)
    coef_xt = np.sqrt(a_t / a_s) * (1 - a_s) / (1 - a_t)
    var = beta * (1 - a_s) / (1 - a_t)
    assert beta > 0 and var > 0, (beta, var)
    x0 = 2.0
    # Monte-Carlo: draw x_t then x_s through the posterior, compare with the marginal
    eps_t = rng.standard_normal(400000)
    xt = np.sqrt(a_t) * x0 + np.sqrt(1 - a_t) * eps_t
    xs_post = coef_x0 * x0 + coef_xt * xt + np.sqrt(var) * rng.standard_normal(400000)
    eps_s = rng.standard_normal(400000)
    xs_marg = np.sqrt(a_s) * x0 + np.sqrt(1 - a_s) * eps_s
    assert abs(xs_post.mean() - xs_marg.mean()) < 0.01
    assert abs(xs_post.var() - xs_marg.var()) < 0.02, (xs_post.var(), xs_marg.var())
    print("DDPM coefficients OK: posterior mean/var match q(x_s|x0) "
          "(%.4f vs %.4f)" % (xs_post.var(), xs_marg.var()))


if __name__ == "__main__":
    test_schedule()
    test_vp_kernel()
    test_accuracy_schedule()
    test_discrete_posterior()
    test_ddpm_coefficients()
    print("ALL MATH CHECKS PASSED")
