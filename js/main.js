;(function () {
	'use strict';

	var header = document.querySelector('.site-header');
	var toggle = document.querySelector('.nav-toggle');
	var mobileNav = document.getElementById('mobile-nav');
	var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

	if (toggle && mobileNav) {
		toggle.addEventListener('click', function () {
			var open = toggle.getAttribute('aria-expanded') === 'true';
			toggle.setAttribute('aria-expanded', String(!open));
			toggle.setAttribute('aria-label', open ? 'Open menu' : 'Close menu');
			if (open) {
				mobileNav.setAttribute('hidden', '');
			} else {
				mobileNav.removeAttribute('hidden');
			}
		});

		mobileNav.querySelectorAll('a').forEach(function (link) {
			link.addEventListener('click', function () {
				toggle.setAttribute('aria-expanded', 'false');
				toggle.setAttribute('aria-label', 'Open menu');
				mobileNav.setAttribute('hidden', '');
			});
		});
	}

	if (header) {
		var onScroll = function () {
			header.classList.toggle('is-scrolled', window.scrollY > 8);
		};
		onScroll();
		window.addEventListener('scroll', onScroll, { passive: true });
	}

	var reveals = document.querySelectorAll('.reveal');
	if (!reveals.length) {
		return;
	}

	if (prefersReducedMotion || !('IntersectionObserver' in window)) {
		reveals.forEach(function (el) {
			el.classList.add('is-visible');
		});
		return;
	}

	var observer = new IntersectionObserver(
		function (entries) {
			entries.forEach(function (entry) {
				if (entry.isIntersecting) {
					entry.target.classList.add('is-visible');
					observer.unobserve(entry.target);
				}
			});
		},
		{ rootMargin: '0px 0px -8% 0px', threshold: 0.12 }
	);

	reveals.forEach(function (el) {
		observer.observe(el);
	});
})();
