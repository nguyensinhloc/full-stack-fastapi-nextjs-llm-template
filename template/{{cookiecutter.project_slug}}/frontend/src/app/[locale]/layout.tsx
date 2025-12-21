{%- if cookiecutter.enable_i18n %}
import { NextIntlClientProvider } from "next-intl";
import { getMessages } from "next-intl/server";
{%- endif %}
import { notFound } from "next/navigation";
import { Providers } from "../providers";
{%- if cookiecutter.enable_i18n %}
import { locales, type Locale } from "@/i18n";
{%- endif %}

{%- if cookiecutter.enable_i18n %}
export function generateStaticParams() {
  return locales.map((locale) => ({ locale }));
}
{%- endif %}

export default async function LocaleLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;

{%- if cookiecutter.enable_i18n %}
  // Validate locale
  if (!locales.includes(locale as Locale)) {
    notFound();
  }

  // Get messages for the current locale
  const messages = await getMessages();

  return (
    <Providers>
      <NextIntlClientProvider messages={messages}>
        {children}
      </NextIntlClientProvider>
    </Providers>
  );
{%- else %}
  // i18n disabled - just render with providers
  return <Providers>{children}</Providers>;
{%- endif %}
}
