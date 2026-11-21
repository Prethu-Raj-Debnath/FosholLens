import { useEffect, useState } from 'react';
import { api, type District } from './api-client';

const STORAGE_KEY = 'fasol-lens.district.v1';

export interface DistrictOption {
  id: string;
  name: string;
}

/**
 * Remember the chosen district across visits.
 *
 * One tap, no permission dialog, works indoors - and coarse enough that the
 * location never becomes identifying. Geolocation stays out on purpose.
 */
export function useDistrict() {
  const [district, setDistrict] = useState<string>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) ?? '';
    } catch {
      return '';
    }
  });

  const [districts, setDistricts] = useState<DistrictOption[]>([]);

  useEffect(() => {
    let cancelled = false;
    api
      .listDistricts()
      .then((list: District[]) => {
        if (!cancelled) setDistricts(list.map(({ id, name }) => ({ id, name })));
      })
      .catch(() => {
        // Leave the selector empty; weather is optional.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function choose(next: string) {
    setDistrict(next);
    try {
      if (next) localStorage.setItem(STORAGE_KEY, next);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Private mode: the in-memory value still works for this session.
    }
  }

  return { district, districts, choose };
}
